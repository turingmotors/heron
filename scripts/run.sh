#!/bin/bash
export WANDB_PROJECT=heron

# llama 7b
# projection + lora
export WANDB_NAME=opt/exp001
deepspeed train.py --config_file projects/$WANDB_NAME.yml
