#!/bin/bash

WANDB_PROJECT_NAME=project-name

MODEL_CONFIG=./projects/opt/exp001.yml
OUTPUT_PATH=./output/llava-bench-ja
QUESTION_PATH=./playground/data/llava-bench-ja/qa90_questions_ja.jsonl

python heron/eval/inference_llava_bench.py\
    --config_file $MODEL_CONFIG\
    --questions_path $QUESTION_PATH\
    --img_root ./playground/data/llava-bench-ja/val2014\
    --output_path $OUTPUT_PATH \
    --device 0\
    --verbose True\
    is_upload_result

EXP_NAME=$(basename "${MODEL_CONFIG}" | cut -d'/' -f 5 | cut -d'.' -f 1)

CONTEXT_PATH="./playground/data/llava-bench-ja/captions_boxes_coco2014_val_80.jsonl"
ANSWER_LIST_PATHS="./playground/data/llava-bench-ja/qa90_gpt4_answers_ja.jsonl ./playground/data/llava-bench-ja/sample_answers.jsonl"
RULE_PATH="./playground/data/llava-bench-ja/rule.json"
OUTPUT_FILE="${OUTPUT_PATH}/${EXP_NAME}_review.json"

python heron/eval/eval_gpt_review_visual.py \
    --question "${QUESTION_PATH}" \
    --context "${CONTEXT_PATH}" \
    --answer-list ${ANSWER_LIST_PATHS} \
    --rule "${RULE_PATH}" \
    --output "${OUTPUT_FILE}"\
    --is_upload_result
