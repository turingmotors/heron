#!/bin/bash

export WANDB_PROJECT_NAME="project-name"

PLAYGROUND_PATH=./playground/data/llava-bench-in-the-wild
MODEL_CONFIG=./projects/video_blip/exp001.yml
OUTPUT_PATH=$PLAYGROUND_PATH/ja/output

# Inference
#python heron/eval/inference_llava_bench_in_the_wild.py \
#    --config_file $MODEL_CONFIG \
#    --questions_path "$PLAYGROUND_PATH/ja/questions_ja.jsonl" \
#    --img_root $PLAYGROUND_PATH/images \
#    --output_path $OUTPUT_PATH \
#    --device 0 \
#    --verbose True \
#    is_upload_result

EXP_NAME=$(basename "${MODEL_CONFIG}" | cut -d'/' -f 5 | cut -d'.' -f 1)

#python heron/eval/remove.py

# Evaluation
QUESTION_PATH="$PLAYGROUND_PATH/ja/questions_ja.jsonl"
CONTEXT_PATH="$PLAYGROUND_PATH/ja/context_ja.jsonl"
ANSWER_LIST_PATHS="$PLAYGROUND_PATH/ja/answers_gpt4_ja.jsonl $OUTPUT_PATH/${EXP_NAME}_answers.jsonl"
RULE_PATH="./playground/data/llava-bench-ja/rule.json"
OUTPUT_FILE="${OUTPUT_PATH}/${EXP_NAME}_reviews.json"

python heron/eval/eval_llava_bench_in_the_wild_review_visual.py \
    --question "${QUESTION_PATH}" \
    --context "${CONTEXT_PATH}" \
    --answer-list ${ANSWER_LIST_PATHS} \
    --rule "${RULE_PATH}" \
    --output "${OUTPUT_FILE}" \
    --is_upload_result
