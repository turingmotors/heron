#!/bin/bash
export WANDB_PROJECT=heron
export PROJECT_NAME=opt/exp002_ds
export WANDB_NAME=$PROJECT_NAME

deepspeed train_ds.py --config_file projects/$PROJECT_NAME.yml
