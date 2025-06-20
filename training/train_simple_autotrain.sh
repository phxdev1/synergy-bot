#!/bin/bash

# Simple AutoTrain CLI script that should work without UI issues

echo "🚀 Corporate Synergy Bot 7B - AutoTrain CLI Training"
echo "===================================================="

# Check if autotrain is installed
if ! command -v autotrain &> /dev/null; then
    echo "Installing AutoTrain..."
    pip install autotrain-advanced
fi

# Login to Hugging Face
echo "Please login to Hugging Face:"
huggingface-cli login

# Run AutoTrain with the fixed dataset
echo "Starting training with AutoTrain CLI..."
autotrain llm \
    --train \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --data-path phxdev/corporate-speak-dataset-autotrain \
    --text-column text \
    --batch-size 4 \
    --epochs 3 \
    --lr 2e-4 \
    --warmup-ratio 0.1 \
    --gradient-accumulation 4 \
    --mixed-precision fp16 \
    --use-peft \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --target-modules "q_proj,k_proj,v_proj,o_proj" \
    --project-name corporate-synergy-bot-7b \
    --push-to-hub \
    --hub-model-id phxdev/corporate-synergy-bot-7b \
    --hub-private false

echo "✅ Training complete!"