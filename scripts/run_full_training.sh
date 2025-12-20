#!/bin/bash
# Full training pipeline for Qwen3 Encoder-Decoder
#
# Usage:
#   ./scripts/run_full_training.sh
#
# Environment variables:
#   MODEL_PATH - Path to initialized model (default: ./initialized-model)
#   OUTPUT_DIR - Output directory (default: ./output/full-training-TIMESTAMP)
#   DATASET - Dataset name (default: HuggingFaceFW/fineweb-edu)
#   PHASE - Training phase: sanity_check, validation, medium, full (default: full)
#   SKIP_SANITY - Skip sanity check (default: false)
#   SKIP_VALIDATION - Skip validation run (default: false)

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-./initialized-model}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/full-training-$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
PHASE="${PHASE:-full}"
SKIP_SANITY="${SKIP_SANITY:-false}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"

# Accelerate config
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp2.yaml}"

echo "=========================================="
echo "Qwen3 Encoder-Decoder Training Pipeline"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Phase: $PHASE"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Sanity Check
if [ "$SKIP_SANITY" != "true" ]; then
    echo ""
    echo "Step 1: Running sanity check..."
    echo "------------------------------------------"

    python scripts/sanity_check.py --model-path "$MODEL_PATH"

    if [ $? -ne 0 ]; then
        echo "ERROR: Sanity check failed!"
        exit 1
    fi
    echo "[PASS] Sanity check passed"
else
    echo ""
    echo "Step 1: Skipping sanity check (SKIP_SANITY=true)"
fi

# Exit early if only running sanity check
if [ "$PHASE" == "sanity_check" ]; then
    echo ""
    echo "=========================================="
    echo "Sanity check complete!"
    echo "=========================================="
    exit 0
fi

# Step 2: Validation Run
if [ "$SKIP_VALIDATION" != "true" ]; then
    echo ""
    echo "Step 2: Running validation (1B tokens)..."
    echo "------------------------------------------"

    # Check if accelerate config exists
    if [ -f "$ACCELERATE_CONFIG" ]; then
        ACCEL_ARGS="--config_file $ACCELERATE_CONFIG"
    else
        echo "Warning: Accelerate config not found at $ACCELERATE_CONFIG"
        echo "Running without config file"
        ACCEL_ARGS=""
    fi

    accelerate launch \
        $ACCEL_ARGS \
        scripts/validation_run.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR/validation" \
        --dataset "$DATASET" \
        --num-steps 5000

    if [ $? -ne 0 ]; then
        echo "ERROR: Validation run failed!"
        exit 1
    fi
    echo "[PASS] Validation passed"
else
    echo ""
    echo "Step 2: Skipping validation (SKIP_VALIDATION=true)"
fi

# Exit early if only running validation
if [ "$PHASE" == "validation" ]; then
    echo ""
    echo "=========================================="
    echo "Validation complete!"
    echo "Results: $OUTPUT_DIR/validation"
    echo "=========================================="
    exit 0
fi

# Step 3: Full Training
echo ""
echo "Step 3: Running full training..."
echo "------------------------------------------"

# Use validation checkpoint if available
if [ -d "$OUTPUT_DIR/validation/model" ]; then
    TRAIN_MODEL="$OUTPUT_DIR/validation/model"
    echo "Using validation checkpoint: $TRAIN_MODEL"
else
    TRAIN_MODEL="$MODEL_PATH"
    echo "Using original model: $TRAIN_MODEL"
fi

# Check if accelerate config exists
if [ -f "$ACCELERATE_CONFIG" ]; then
    ACCEL_ARGS="--config_file $ACCELERATE_CONFIG"
else
    echo "Warning: Accelerate config not found at $ACCELERATE_CONFIG"
    echo "Running without config file"
    ACCEL_ARGS=""
fi

accelerate launch \
    $ACCEL_ARGS \
    scripts/train.py \
    --config configs/training_config.yaml \
    --output_dir "$OUTPUT_DIR/training"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -la "$OUTPUT_DIR"
echo "=========================================="
