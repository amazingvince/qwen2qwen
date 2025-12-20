# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a Qwen3-based encoder-decoder model following the T5Gemma 2 architecture pattern. The goal is to create a bidirectional encoder from Qwen3-0.6B that can be trained using UL2 (Unified Language Learner) denoising objectives and later extracted as a standalone text embedding model.

## Architecture

The model consists of:
- **Encoder**: Qwen3 with bidirectional attention (causal mask removed), using GQA, RoPE, QK-Norm, and RMSNorm
- **Decoder**: Qwen3 with merged self/cross attention following T5Gemma 2's parameter-efficient design
- **Tied Embeddings**: Single embedding matrix shared across encoder input, decoder input, and LM head (~10.5% parameter savings)
- **Sentinel Tokens**: 100 additional tokens for UL2 span corruption training

## Project Structure

```
src/
└── qwen3_encdec/
    ├── __init__.py
    ├── configuration_qwen3_encdec.py   # Qwen3EncoderDecoderConfig (implemented)
    ├── tokenization_qwen3_encdec.py    # Tokenizer with sentinel tokens (implemented)
    ├── modeling_qwen3_encoder.py       # Bidirectional encoder
    ├── modeling_qwen3_decoder.py       # Merged attention decoder
    ├── modeling_qwen3_encdec.py        # Qwen3ForSeq2SeqLM combined model
    └── configs/
        └── default_config.json
tests/
├── __init__.py
├── test_configuration.py               # Configuration tests (implemented)
├── test_tokenization.py                # Tokenizer tests (implemented)
└── test_tokenization_integration.py    # Integration tests with real Qwen3
scripts/
├── initialize_from_qwen3.py            # Weight initialization
├── train.py                            # Training script
└── extract_encoder.py                  # Encoder extraction
```

## Key Commands

### Installation
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
pytest tests/ -v
pytest tests/test_configuration.py -v --cov=qwen3_encdec
```

### Training (planned)
```bash
# Launch distributed training with FSDP2
accelerate launch --config_file configs/accelerate_fsdp2.yaml scripts/train.py --config configs/training_config.yaml
```

### Data Pipeline
```bash
# Prepare and test UL2 data pipeline
python scripts/prepare_ul2_data.py --tokenizer ./qwen3-encdec-tokenizer --analyze-only
```

## UL2 Training Tasks

The training uses 5 denoising tasks with weights 1:1:1:1:4:
- **R-Denoiser 1**: mean_span=3, corruption=0.15 (short spans, low corruption)
- **R-Denoiser 2**: mean_span=12, corruption=0.50 (medium spans, high corruption)
- **X-Denoiser 1**: mean_span=32, corruption=0.15 (long spans, low corruption)
- **X-Denoiser 2**: mean_span=32, corruption=0.50 (long spans, high corruption)
- **S-Denoiser**: prefix-to-suffix with 75% as target (4x weight, teaches causal generation)

## Key Implementation Details

1. **Qwen3 vs Encoder**: The only difference is removing the causal mask for bidirectional attention
2. **Merged Attention**: Decoder uses single attention module for both self and cross attention
3. **Vocab Size**: Original Qwen3 (151,936) + 100 sentinel tokens = 152,036
4. **GQA Configuration**: 16 query heads, 8 KV heads (Qwen3-0.6B default)

## Dependencies

- transformers>=4.40.0
- torch>=2.0.0
- accelerate (for FSDP2 distributed training)
- datasets (for streaming data pipelines)
- pytest (for testing)

## Implementation Stories

The `plan/` directory contains detailed implementation stories (01-11) covering:
1. Project setup and configuration
2. Tokenizer extension with sentinel tokens
3. Bidirectional encoder implementation
4. Merged attention decoder
5. Combined seq2seq model
6. Weight initialization from Qwen3
7. UL2 data pipeline
8. Training infrastructure (FSDP2/DeepSpeed)
9. Training execution
10. Encoder extraction
11. Evaluation
