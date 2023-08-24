import glob
import os
from base64 import b64decode
from io import BytesIO
from typing import Any, Optional, Union

import deepspeed
import fire
import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
)

from heron.models.utils import load_model, load_pretrained_weight, apply_lora_model, set_trainable_params
from heron.datasets.utils import get_dataset

GitLLMForCausalLM = Any


def main(config_file: str, local_rank: int = 0):
    # get config
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)

    if os.environ["WANDB_NAME"] is not None:
        config["training"]["output_dir"] = os.path.join(
            config["training"]["output_dir"], os.environ["WANDB_NAME"]
        )

    # distributed learning
    deepspeed.init_distributed()

    # model
    model_name = config["settings"]["model_name"]
    vision_model_name = config["settings"]["vision_model_name"]
    num_image_with_embedding = config["settings"]["num_image_with_embedding"]
    is_fp16 = config["training"]["fp16"]

    # configの割り当て
    max_length = config["settings"]["max_length"]
    keys_finetune = config["settings"]["keys_finetune"]

    # DatasetのLoad
    train_dataset, val_dataset = get_dataset(config, model_name, vision_model_name, max_length)

    # 訓練に関するconfig
    training_args = TrainingArguments(**config["training"])

    # load model
    model = load_model(model_name, vision_model_name, num_image_with_embedding, is_fp16)

    # lora
    if config["use_lora"]:
        keys_finetune.append("lora")
        model = apply_lora_model(model, model_name, config)

    # load pretrained weight
    if config["settings"]["load_pretrained"] is not None:
        load_pretrained_weight(model, config["settings"]["load_pretrained"])
        print(
            f'Successfully loading pretrained weights from {config["settings"]["load_pretrained"]}'
        )

    # Set trainable params
    set_trainable_params(model, model_name, keys_finetune)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    with torch.autocast("cuda"):
        result = trainer.train()

    # Save the finel checkpoint
    # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L2281
    final_save_path = os.path.join(
        config["training"]["output_dir"], os.environ["WANDB_NAME"] + "_final"
    )
    trainer.save_pretrained(final_save_path)
    if "zero3" in config["training"]["deepspeed"]:
        # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
        # config `stage3_gather_16bit_weights_on_model_save` is True
        trainer.model_wrapped.save_checkpoint(final_save_path)


if __name__ == "__main__":
    fire.Fire(main)
