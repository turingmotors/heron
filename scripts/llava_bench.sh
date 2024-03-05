#!/bin/bash

export WANDB_PROJECT_NAME="project-name"

PLAYGROUND_PATH=./playground/data/llava-bench-ja
MODEL_CONFIG=./projects/video_blip/exp001.yml
OUTPUT_PATH=$PLAYGROUND_PATH/output

# Inference
python heron/eval/inference_llava_bench.py \
    --config_file $MODEL_CONFIG \
    --questions_path "$PLAYGROUND_PATH/qa90_questions_ja.jsonl" \
    --img_root $PLAYGROUND_PATH/val2014 \
    --output_path $OUTPUT_PATH \
    --device 0 \
    --verbose True \
    is_upload_result

EXP_NAME=$(basename "${MODEL_CONFIG}" | cut -d'/' -f 5 | cut -d'.' -f 1)

# Evaluation
QUESTION_PATH="$PLAYGROUND_PATH/qa90_questions_ja.jsonl"
CONTEXT_PATH="$PLAYGROUND_PATH/captions_boxes_coco2014_val_80.jsonl"
ANSWER_LIST_PATHS="$PLAYGROUND_PATH/qa90_gpt4_answers_ja.jsonl $OUTPUT_PATH/${EXP_NAME}_answers.jsonl"
RULE_PATH="$PLAYGROUND_PATH/rule.json"
OUTPUT_FILE="${OUTPUT_PATH}/${EXP_NAME}_reviews.json"

python heron/eval/eval_gpt_review_visual.py \
    --question "${QUESTION_PATH}" \
    --context "${CONTEXT_PATH}" \
    --answer-list ${ANSWER_LIST_PATHS} \
    --rule "${RULE_PATH}" \
    --output "${OUTPUT_FILE}" \
    --is_upload_result
