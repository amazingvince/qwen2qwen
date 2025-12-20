#!/bin/bash
# Launch training for Qwen3 Encoder-Decoder
#
# Usage:
#   ./scripts/launch_training.sh
#   ./scripts/launch_training.sh --dummy_data  # For testing

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export WANDB_PROJECT=${WANDB_PROJECT:-"qwen3-encoder-decoder"}

# Accelerate configuration (FSDP2 by default)
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"configs/accelerate_fsdp2.yaml"}

# Training configuration
TRAIN_CONFIG=${TRAIN_CONFIG:-"configs/training_config.yaml"}

# Output directory with timestamp
OUTPUT_DIR=${OUTPUT_DIR:-"./output/qwen3-encdec-$(date +%Y%m%d_%H%M%S)"}

# Print configuration
echo "============================================"
echo "Qwen3 Encoder-Decoder Training"
echo "============================================"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "ACCELERATE_CONFIG: $ACCELERATE_CONFIG"
echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "============================================"

# Check if accelerate config exists
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "Warning: Accelerate config not found at $ACCELERATE_CONFIG"
    echo "Running without config file (will use defaults)"
    ACCELERATE_ARGS=""
else
    ACCELERATE_ARGS="--config_file $ACCELERATE_CONFIG"
fi

# Check if training config exists
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "Error: Training config not found at $TRAIN_CONFIG"
    exit 1
fi

# Launch training
accelerate launch \
    $ACCELERATE_ARGS \
    scripts/train.py \
    --config "$TRAIN_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo "============================================"
echo "Training complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"
