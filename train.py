import os
from typing import Any

import deepspeed
import fire
import torch
import yaml
from transformers import Trainer, TrainingArguments

from heron.datasets.utils import get_dataset
from heron.models.utils import (
    apply_lora_model,
    load_model,
    load_pretrained_weight,
    set_trainable_params,
)

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

    # configの割り当て
    keys_finetune = config["settings"]["keys_finetune"]

    # DatasetのLoad
    train_dataset, val_dataset = get_dataset(config)

    # 訓練に関するconfig
    training_args = TrainingArguments(**config["training"])

    # load model
    model = load_model(config)

    # lora
    if config["use_lora"]:
        keys_finetune.append("lora")
        model = apply_lora_model(model, config)

    # load pretrained weight
    if config["settings"]["load_pretrained"] is not None:
        load_pretrained_weight(model, config["settings"]["load_pretrained"])
        print(
            f'Successfully loading pretrained weights from {config["settings"]["load_pretrained"]}'
        )

    # Set trainable params
    set_trainable_params(model, keys_finetune)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    with torch.autocast("cuda"):
        trainer.train()

    # Save the finel checkpoint
    final_save_path = os.path.join(
        config["training"]["output_dir"], os.environ["WANDB_NAME"] + "_final"
    )
    trainer.save_model(final_save_path)


if __name__ == "__main__":
    fire.Fire(main)
