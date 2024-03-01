#!/bin/bash

WANDB_PROJECT_NAME=project-name

MODEL_CONFIG=./projects/opt/exp001.yml
OUTPUT_PATH=./eval/llava-bench-ja/results
QUESTION_PATH=./eval/llava-bench-ja/qa90_questions_ja.jsonl

python eval/llava-bench-ja/inference_llava_bench.py\
    --config_file $MODEL_CONFIG\
    --questions_path $QUESTION_PATH\
    --img_root ./eval/llava-bench-ja/datasets/val2014\
    --output_path $OUTPUT_PATH \
    --device 0\
    --verbose True\
    is_upload_result

EXP_NAME=$(basename "${MODEL_CONFIG}" | cut -d'/' -f 5 | cut -d'.' -f 1)

CONTEXT_PATH="./eval/llava-bench-ja/caps_boxes_coco2014_val_80.jsonl"
ANSWER_LIST_PATHS="./eval/llava-bench-ja/qa90_gpt4_answer_ja_v2.jsonl ./eval/llava-bench-ja/sample_answer.jsonl"
RULE_PATH="./eval/llava-bench-ja/rule.json"
OUTPUT_FILE="${OUTPUT_PATH}/${EXP_NAME}_review.json"

python eval/llava-bench-ja/eval_gpt_review_visual.py \
    --question "${QUESTION_PATH}" \
    --context "${CONTEXT_PATH}" \
    --answer-list ${ANSWER_LIST_PATHS} \
    --rule "${RULE_PATH}" \
    --output "${OUTPUT_FILE}"\
    --is_upload_result
