#!/bin/bash
export WANDB_PROJECT=git_llm

# llama 7b
# projection + lora
export WANDB_NAME=exp050_llama
deepspeed train.py --config_file configs/training_config_$WANDB_NAME.yml
