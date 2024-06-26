#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="Qwen/Qwen-7B-Chat-Int4" # Set the path if you do not want to load from huggingface directly
#MODEL="/home/eawlot3000/cherish/Qwen/finetune/output_qwen/checkpoint-1000"

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
#DATA="path_to_data" # og

DATA="/home/eawlot3000/cherish/fine_tune_chatbot_with_ellie/data/qwen_lv3_data_json/2_updated_renamed_formatted_result.json"

function usage() {
#    echo '
#Usage: bash finetune/finetune_lora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
#'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=0

python3 ../finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --bf16 False \
  --output_dir output_qwen \
  --num_train_epochs 2000 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora


# !!!!
# i have turn that shit off!

# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json
