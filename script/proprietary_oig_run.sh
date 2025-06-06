#!/bin/bash

open_model_list=(
    "gpt-4o-mini" # 0
    "gpt-4o" # 1
    "claude-3-5-haiku-20241022" # 2
    "claude-3-5-sonnet-20241022" # 3
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

python ../src/proprietary_oig.py \
    --model_name "$selected_model" \
    --task "oig" \
    --seed 42

