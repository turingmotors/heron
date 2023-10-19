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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data import (
    DataCollatorPadToMaxLen,
    build_dataset,
    shuffle_dataset,
    split_dataset,
)
from utils.ds_utils import get_train_ds_config
from utils.model import create_dsvl_model_and_transforms
from utils.module.lora import (
    convert_linear_layer_to_lora,
    fuse_lora,
    only_optimize_lora_parameters,
    unfuse_lora,
)
from utils.utils import (
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    print_rank_0,
    save_hf_format,
    save_zero_three_model,
    set_random_seed,
    to_device,
)

# import heron library
sys.path.append("/home/yuma_ochi/heron-exp")
from heron.datasets.utils import get_dataset
from heron.models.utils import (
    apply_lora_model,
    load_model,
    load_pretrained_weight,
    set_trainable_params,
    unload_and_merge_lora,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a multi-modal task"
    )

    parser.add_argument(
        "--data_path", type=str, default="./data/", help="Where the training data are stored."
    )

    parser.add_argument(
        "--data_debug_path",
        type=str,
        default=None,
        help="If provided, will save 10 training samples" "to the path for debug purpose.",
    )

    parser.add_argument(
        "--data_train_split_ratio",
        type=float,
        default=0.9,
        help="Ratio of dataset to be splitted as train data. The remaining becomes eval data.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=["minigpt4"],
        help="Name of training dataset(s) to be used. Accepted format:"
        "1) a single dataset name, 2) multiple dataset names in the"
        "form: dataset1 dataset2 ...",
    )

    parser.add_argument(
        "--dataset_samples",
        nargs="*",
        default=["all"],
        help="How many samples do we use from each dataset."
        "Should be either a integer number or string all which"
        "means use all samples. For example: all 512 means"
        "using all samples form first data and 512 samples"
        "from second data",
    )

    parser.add_argument(
        "--dataset_concatenate_samples",
        nargs="*",
        default=[1],
        help="How many samples do we concatenate from each dataset."
        "Should be either a integer number or string. 1 which"
        "means use 1 sample for each datapoint",
    )

    parser.add_argument(
        "--max_num_image_per_sample",
        type=int,
        default=8,
        help="The maximum number of images per sample.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="The maximum sequence length, note that image tokens are included.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_pretraining_components",
        type=float,
        default=0,
        help="Initial learning rate for pre-trained weight, e.g., embedding (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=6,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=0,
        help="Number of steps (>1) or ratios (<=1) for the warmup in the lr scheduler.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--lm_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--vision_model_name_or_path", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument(
        "--enable_mmca_attention",
        action="store_true",
        help="enable the new proposed attn, which is similar to cross attention",
    )
    parser.add_argument(
        "--vis_proj",
        type=str,
        default="baseline",
        help="[baseline, vit, or perceiver], used to projection vision feature to LLM embedding",
    )
    # deepspeed features
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="FP16 or BF16 precision. FP16 is recommended for typical use cases. BF16 is good for large models",
    )
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lang_lora_dim",
        type=int,
        default=0,
        help="Use LoRA for fine-tuning language decoder (> 0).",
    )
    parser.add_argument(
        "--lang_lora_module_name",
        type=str,
        default="model.layers.",
        help="The scope name of the target LoRA parameters.",
    )
    parser.add_argument(
        "--vis_lora_dim",
        type=int,
        default=0,
        help="Use LoRA for fine-tuning visual encoder (> 0).",
    )
    parser.add_argument(
        "--vis_lora_module_name",
        type=str,
        default="encoder.layers.",
        help="The scope name of the target LoRA parameters.",
    )
    parser.add_argument(
        "--only_optimize_lora", action="store_true", help="Only optimize the LoRA parameters."
    )
    parser.add_argument("--heron_config_file", type=str, help="heronのconfigファイルパス")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.learning_rate_pretraining_components == 0.0:
        # if we do not provide special learning rate, mainly for embedding, the same lr is applied
        args.learning_rate_pretraining_components = args.learning_rate
    assert args.num_warmup_steps >= 0, "--num_warmup_steps must be >= 0"
    if "qwen" in args.vision_model_name_or_path.lower():
        assert (
            args.vis_proj == "baseline"
        ), "qwen's model only support baseline vis_proj as it has the perceiver module inside"
    return args


def main(config_file: str, local_rank: int = 0):
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)
        model_config = config["model_config"]
        training_config = config["training_config"]

    if os.environ.get("WANDB_NAME") is not None:
        training_config["output_dir"] = os.path.join(
            training_config["output_dir"], os.environ["WANDB_NAME"]
        )

    # args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(args, offload=False, stage=args.zero_stage)
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    # Heronのconfigを読み出す
    with open(args.heron_config_file, "r") as i_:
        config = yaml.safe_load(i_)
        model_config = config["model_config"]
        training_config = config["training_config"]

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()
    """ ↓ Heronのモデル定義 """
    # load model
    model = load_model(model_config)

    if model_config["use_lora"]:
        model = apply_lora_model(model, model_config)
    """ ↑ Heronのモデル定義 """

    print_rank_0(model, args.global_rank)

    """ ↓ Heron """
    config["dataset_config_path"] = [
        os.path.join("/home/yuma_ochi/heron-exp", path) for path in config["dataset_config_path"]
    ]
    train_dataset, eval_dataset = get_dataset(config)
    """ ↑ Heron """

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=DistributedSampler(eval_dataset, shuffle=False),
    )

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, small_lr=args.learning_rate_pretraining_components
    )

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.num_warmup_steps <= 1:
        args.num_warmup_steps = int(
            args.num_warmup_steps * args.num_train_epochs * num_update_steps_per_epoch
        )
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    # assert 0

    start_epoch = 0
    """ 学習済みモデルは使わない """
    # # let load checkpoint
    # if os.path.exists(os.path.join(args.output_dir, 'latest')):
    #     # we have the deepspeed chekpoint so it is a resumed job
    #     # TODO: after loading the ckpt, the global step is not loaded. Need to ask Tunji/Ammar for help.
    #     _, client_state = model.load_checkpoint(args.output_dir)
    #     start_epoch = client_state['epoch']
    #     best_loss = client_state['best_loss']
    #     random.setstate(client_state['random_rng_state'])
    #     np.random.set_state(client_state['np_rng_state'])
    #     torch.set_rng_state(client_state['torch_rng_state'])
    #     torch.cuda.set_rng_state(client_state['torch_cuda_rng_state'])

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def evaluation(model, eval_dataloader):
        model.eval()
        print("Evaluation")
        acc_loss = 0
        for step, batch in enumerate(eval_dataloader):
            print(step)
            with torch.no_grad():
                batch = to_device(batch, device)
                loss = model(
                    batch["image"].half(),
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    input_labels=batch["labels"],
                    image_num=batch["image_num"],
                )[0]
            acc_loss += loss
        model.train()
        acc_loss = get_all_reduce_mean(acc_loss).item()
        ave_loss = acc_loss / (step + 1)
        print_rank_0(f"the eval average_loss: {ave_loss}", args.global_rank)
        return ave_loss

    # Train!
    if start_epoch == 0:
        print_rank_0("***** Before training *****", args.global_rank)
        # evaluation(model, eval_dataloader)
        best_loss = 1e6

    print_rank_0("***** Running training *****", args.global_rank)
    for epoch in range(start_epoch, args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()
        acc_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(
                batch, device
            )  # torch.size(1, 3, 224, 224]) #torch.Size([1, 1, 3, 224, 224])

            """ ↓ Heronの1step """
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            # position_ids = batch["position_ids"]
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )[0]
            print(loss)

            """ ↑ Heronの1step """

            acc_loss += loss.detach().clone()
            model.backward(loss)
            model.step()
        model.tput_timer.update_epoch_count()
        acc_loss = get_all_reduce_mean(acc_loss).item()
        print_rank_0(f"Epoch {epoch+1}, the average_loss: {acc_loss/step}", args.global_rank)
        eval_loss = evaluation(model, eval_dataloader)

        if eval_loss < best_loss:
            best_loss = eval_loss

        model = fuse_lora(model)
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, f"epoch-{epoch}")
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(
                model,
                args.global_rank,
                args.output_dir,
                zero_stage=args.zero_stage,
                sub_folder=f"epoch-{epoch}",
            )
        model = unfuse_lora(model)
        # save deepspeed zero checkpoint so we can resume training if needed
        client_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch": epoch + 1,  # start from next epoch
            "best_loss": best_loss,
        }
        model.save_checkpoint(args.output_dir, client_state=client_state)  # save to the latest


if __name__ == "__main__":
    fire.Fire(main)
