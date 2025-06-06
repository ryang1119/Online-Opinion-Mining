#!/bin/bash

open_model_list=(
    "meta-llama/Meta-Llama-3-8B-Instruct" # 0
    "meta-llama/Meta-Llama-3-70B-Instruct" # 1
    "google/gemma-2-9b-it" # 2
    "google/gemma-2-27b-it" # 3
    "Qwen/Qwen2.5-7B-Instruct" # 4
    "deepseek-ai/deepseek-llm-7b-chat" # 5
)

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_index>"
    echo "Available indices:"
    for i in "${!open_model_list[@]}"; do
        echo "$i: ${open_model_list[$i]}"
    done
    exit 1
fi

model_index=$1

if ! [[ "$model_index" =~ ^[0-9]+$ ]]; then
    echo "Error: model_index must be a number."
    exit 1
fi

if [ "$model_index" -lt 0 ] || [ "$model_index" -ge "${#open_model_list[@]}" ]; then
    echo "Error: model_index out of range. Choose a number between 0 and $((${#open_model_list[@]} - 1))."
    exit 1
fi

selected_model="${open_model_list[$model_index]}"

echo "Running model: $selected_model"

python ../src/open_sourced_oig.py \
    --model_name "$selected_model" \
    --port "8000" \
    --task "oig" \
    --seed 42

