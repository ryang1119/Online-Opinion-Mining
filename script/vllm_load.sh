#!/bin/bash

open_model_list=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3-70B-Instruct"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "Qwen/Qwen2.5-7B-Instruct"
    "deepseek-ai/deepseek-llm-7b-chat"
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

export CUDA_VISIBLE_DEVICES=3

echo "Running model: $selected_model"

python -m vllm.entrypoints.openai.api_server \
    --model "$selected_model" \
    --tensor-parallel-size 1 \
    --gpu_memory_utilization 0.95 \
    --seed 42 \
    --port 8000
