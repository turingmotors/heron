#!/bin/bash
export WANDB_PROJECT=heron
export PROJECT_NAME=llama/exp001
export WANDB_NAME=$PROJECT_NAME

deepspeed train.py --config_file projects/$PROJECT_NAME.yml
