#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import os
import random
import sys

import deepspeed
import fire
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, AutoTokenizer, SchedulerType, get_scheduler

import wandb
from heron.datasets.utils import get_dataset
from heron.models.utils import (
    apply_lora_model,
    load_model,
    load_pretrained_weight,
    set_trainable_params,
    unload_and_merge_lora,
)
from heron.utils.ds_utils import get_train_ds_config
from heron.utils.utils import (
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    print_rank_0,
    save_hf_format,
    save_zero_three_model,
    set_random_seed,
    to_device,
)


def main(config_file: str, local_rank: int = 0):
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)
        model_config = config["model_config"]
        training_config = config["training_config"]

    if os.environ.get("WANDB_NAME") is not None:
        training_config["output_dir"] = os.path.join(
            training_config["output_dir"], os.environ["WANDB_NAME"]
        )

    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    training_config["global_rank"] = torch.distributed.get_rank()

    set_random_seed(training_config["seed"])

    # DeepSpeedの初期化に必要な変数を設定
    ds_config = get_train_ds_config(
        training_config,
        offload=training_config["cpu_offload"],
        stage=training_config["zero_stage"],
    )
    ds_config["train_micro_batch_size_per_gpu"] = training_config["per_device_train_batch_size"]
    ds_config["train_batch_size"] = (
        training_config["per_device_train_batch_size"]
        * torch.distributed.get_world_size()
        * training_config["gradient_accumulation_steps"]
    )
    # wandb の初期化
    if os.environ.get("WANDB_NAME") is not None and local_rank == 0:
        wandb.init(project=os.environ["WANDB_PROJECT"], config=config)

    # すべてのプロセスの処理が終わるまで待機
    torch.distributed.barrier()

    # load model
    model = load_model(model_config)

    if model_config["use_lora"]:
        # VisualChatのLoRA実装 (w/o peft)
        # model = convert_linear_layer_to_lora(model, ["query_key_value"], lora_dim=8)

        # HeronのLoRA実装 (w/ peft)
        model = apply_lora_model(model, model_config)

    print_rank_0(model, training_config["global_rank"])

    # datasetの読み込み
    train_dataset, eval_dataset = get_dataset(config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["per_device_train_batch_size"],
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
        num_workers=training_config["dataloader_num_workers"],
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config["per_device_eval_batch_size"],
        sampler=DistributedSampler(eval_dataset, shuffle=False),
        num_workers=training_config["dataloader_num_workers"],
    )

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,
        training_config["weight_decay"],
        small_lr=training_config["learning_rate_pretraining_components"],
    )

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=training_config["learning_rate"], betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_config["gradient_accumulation_steps"]
    )
    if training_config["num_warmup_steps"] <= 1:
        training_config["num_warmup_steps"] = int(
            training_config["num_warmup_steps"]
            * training_config["num_train_epochs"]
            * num_update_steps_per_epoch
        )
    else:
        training_config["num_warmup_steps"] = int(training_config["num_warmup_steps"])
    lr_scheduler = get_scheduler(
        name=training_config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=training_config["num_warmup_steps"],
        num_training_steps=training_config["num_train_epochs"] * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    start_epoch = 0
    # let load checkpoint
    if os.path.exists(os.path.join(training_config["output_dir"], "latest")):
        # we have the deepspeed chekpoint so it is a resumed job
        # TODO: after loading the ckpt, the global step is not loaded. Need to ask Tunji/Ammar for help.
        _, client_state = model.load_checkpoint(training_config["output_dir"])
        start_epoch = client_state["epoch"]
        best_loss = client_state["best_loss"]
        random.setstate(client_state["random_rng_state"])
        np.random.set_state(client_state["np_rng_state"])
        torch.set_rng_state(client_state["torch_rng_state"])
        torch.cuda.set_rng_state(client_state["torch_cuda_rng_state"])

    if training_config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    def evaluation(model, eval_dataloader):
        model.eval()
        print_rank_0("***** Evaluation *****", training_config["global_rank"])
        acc_loss = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = to_device(batch, device)
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"].half(),
                    labels=batch["labels"],
                )[0]
            acc_loss += loss
        model.train()
        acc_loss = get_all_reduce_mean(acc_loss).item()
        ave_loss = acc_loss / (step + 1)
        print_rank_0(f"the eval average_loss: {ave_loss}", training_config["global_rank"])
        return ave_loss

    # Train!
    if start_epoch == 0:
        print_rank_0("***** Before training *****", training_config["global_rank"])
        # evaluation(model, eval_dataloader)
        best_loss = 1e6

    print_rank_0("***** Running training *****", training_config["global_rank"])
    for epoch in range(start_epoch, training_config["num_train_epochs"]):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{training_config['num_train_epochs']}, Total Micro Batches {len(train_dataloader)}",
            training_config["global_rank"],
        )
        model.train()
        acc_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)

            # ここはDatasetの出力とモデルのforward関数を参考にした
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pixel_values = batch["pixel_values"].half()
            labels = batch["labels"]
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )[0]

            acc_loss += loss.detach().clone()
            model.backward(loss)
            # この中でgradient accumulationが行われることに注意
            model.step()

            # wandbへのlog
            if os.environ.get("WANDB_NAME") is not None and local_rank == 0:
                wandb.log(
                    {
                        "Train/epoch": epoch,
                        "Train/step": step,
                        "Train/loss": loss.detach(),
                        "Train/average_loss": acc_loss / step,
                    }
                )

        model.tput_timer.update_epoch_count()
        acc_loss = get_all_reduce_mean(acc_loss).item()
        print_rank_0(
            f"Epoch {epoch+1}, the average_loss: {acc_loss/step}", training_config["global_rank"]
        )
        eval_loss = evaluation(model, eval_dataloader)

        if eval_loss < best_loss:
            best_loss = eval_loss

        # wandbへのlog
        if os.environ.get("WANDB_NAME") is not None and local_rank == 0:
            wandb.log(
                {
                    "Eval/loss": eval_loss,
                }
            )

        # 途中のチェックポイントの保存
        client_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch": epoch + 1,  # start from next epoch
            "best_loss": best_loss,
        }
        model.save_checkpoint(
            training_config["output_dir"], client_state=client_state
        )  # save to the latest

        # モデルの保存(LoRAをモデルにマージしたもの)
        # if model_config["use_lora"]:
        #     model = unload_and_merge_lora(model, model_config)

        if training_config["global_rank"] == 0:
            save_hf_format(model, training_config)
        if training_config["zero_stage"] == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(
                model,
                training_config["global_rank"],
                training_config["output_dir"],
                zero_stage=training_config["zero_stage"],
                sub_folder=f"epoch-{epoch}",
            )


if __name__ == "__main__":
    fire.Fire(main)
