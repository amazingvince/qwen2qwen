# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-based encoder-decoder following the T5Gemma 2 architecture pattern. Converts Qwen3-0.6B into a bidirectional encoder trained with UL2 denoising objectives, then extracted as a standalone text embedding model.

## Environment

Use the `misc` conda environment for all operations. GPU (RTX 5090, bf16/tf32 supported) available.

## Key Commands

### Installation
```bash
pip install -e ".[dev]"                    # Development (pytest)
pip install -e ".[dev,training]"           # Training (includes UL2_5)
pip install -e ".[optimizations]"          # Liger kernels + cut-cross-entropy
pip install -e ".[all]"                    # Everything
```

### Linting (required before commits)
```bash
ruff check --fix . && ruff format .
```

### Testing
```bash
pytest tests/ -v                           # All tests
pytest tests/test_encoder.py -v            # Single file
pytest tests/test_encoder.py::test_forward -v  # Single test
pytest tests/ -v --cov=src/qwen3_encdec    # With coverage
```

### Training
```bash
# Single GPU (for development/testing)
python scripts/train.py --config configs/training_single_gpu.yaml

# Multi-GPU with FSDP2
accelerate launch --config_file configs/accelerate_fsdp2.yaml \
    scripts/train.py --config configs/training_config.yaml
```

### Model Initialization
```bash
python scripts/initialize_model.py \
    --qwen3-model Qwen/Qwen3-0.6B \
    --output-dir ./qwen3-encdec-initialized \
    --verify
```

### Encoder Extraction
```bash
python scripts/extract_encoder.py \
    --checkpoint ./checkpoints/checkpoint-100000 \
    --output ./extracted_encoder \
    --pooling_mode mean \
    --export_sentence_transformers
```

### Evaluation
```bash
python scripts/quick_eval.py --encoder_path ./extracted_encoder  # Quick sanity check
python scripts/run_evaluation.py --encoder_path ./extracted_encoder --run_sts --run_mteb
```

## Architecture

**Encoder-Decoder (training):**
- Encoder: Qwen3 with bidirectional attention (causal mask removed), 28 layers, GQA (16/8 heads)
- Decoder: Merged self/cross attention (T5Gemma 2 pattern), 28 layers
- Tied embeddings: Shared across encoder input, decoder input, LM head (~10.5% savings)
- Vocab: 151,936 (Qwen3) + 100 sentinel tokens = 152,036

**Extracted Encoder (inference):**
- `Qwen3StandaloneEncoderModel` with pooler (mean/cls/last/weighted_mean)
- Compatible with sentence-transformers

## Project Structure

- `src/qwen3_encdec/` - Core model: config, tokenizer, encoder, decoder, seq2seq, `encoder_only.py` (standalone encoder for inference)
- `src/data/` - UL2_5-backed data pipeline via `UL2DataCollator`
- `src/training/` - Config, trainer, monitoring, memory utils, `optimizations.py` (Liger kernels, CCE, TF32)
- `src/extraction/` - Encoder extraction, checkpoint averaging, sentence-transformers export
- `src/evaluation/` - MTEB, STS, retrieval benchmarks
- `scripts/` - CLI entry points (train.py, extract_encoder.py, etc.)
- `configs/` - YAML training configs

## UL2 Training Tasks (T5Gemma 2 mixture, weights 1:1:1:1:4)

| Task | Mean Span | Corruption | Purpose |
|------|-----------|------------|---------|
| R1 | 3 | 15% | Short spans, low corruption |
| R2 | 12 | 50% | Medium spans, high corruption |
| X1 | 32 | 15% | Long spans, low corruption |
| X2 | 32 | 50% | Long spans, high corruption |
| S | prefix | 75% | Prefix-to-suffix (4x weight, teaches causal generation) |

## GPU Optimizations

Training configs enable by default:
- **BF16**: Always use bf16, never fp16
- **TF32**: 3x faster matmuls on Ampere+ GPUs
- **Liger kernels**: Optimized RMSNorm, SwiGLU MLP (optional dep)
- **Cut Cross Entropy**: 24GB → 1MB memory for loss computation (optional dep)
- **Fused AdamW**: CUDA-accelerated optimizer
- **Gradient checkpointing**: Enabled for memory efficiency
- **torch.compile**: Disabled by default (causes issues with dynamic shapes in RoPE)

## Key Implementation Details

1. **Bidirectional encoder**: Only difference from Qwen3 is removing the causal mask
2. **Merged attention**: Decoder uses single attention module for both self and cross attention
3. **UL2_5 integration**: Data collation via `UL2DataCollator` adapting the UL2_5 library
4. **Config-driven training**: All training params in YAML files under `configs/`
5. **Streaming data**: Uses HuggingFace datasets streaming mode
6. **FSDP2**: Auto-wraps `Qwen3EncoderLayer` and `Qwen3DecoderLayer`

## Implementation Stories

Detailed design docs in `plan/` directory (01-11) covering each component's implementation rationale.

## Development Workflow

- After code changes, always run `ruff check --fix . && ruff format .`
- After code changes, run relevant tests
- When signatures change, update type hints and docstrings
- Commits should be atomic; conventional commit messages preferred
- Work typically done in branches, squash merged to main
