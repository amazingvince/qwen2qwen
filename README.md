# Qwen3 Encoder-Decoder

A Qwen3-based encoder-decoder model following the T5Gemma 2 architecture pattern. This project converts Qwen3-0.6B into a bidirectional encoder that can be trained using UL2 denoising objectives and extracted as a standalone text embedding model.

---

- [Qwen3 Encoder-Decoder](#qwen3-encoder-decoder)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Training](#training)
  - [Encoder Extraction](#encoder-extraction)
  - [Evaluation](#evaluation)
  - [Project Structure](#project-structure)
  - [Testing](#testing)
  - [Model Details](#model-details)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)

---

## Overview

The key insight from T5Gemma 2 is that you can take a pretrained decoder-only LLM and convert it into a powerful encoder-decoder by:

1. Making the encoder bidirectional (removing the causal mask)
2. Training with UL2 span corruption objectives
3. Extracting just the encoder for embedding tasks

This project implements this approach for Qwen3-0.6B, creating a ~300M parameter bidirectional encoder suitable for text embeddings, semantic search, and retrieval tasks.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3ForSeq2SeqLM                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │     Encoder     │         │        Decoder          │   │
│  │  (Bidirectional)│         │  (Merged Self+Cross)    │   │
│  │                 │         │                         │   │
│  │  - GQA (16/8)   │────────▶│  - GQA (16/8)          │   │
│  │  - RoPE         │ encoder │  - RoPE                 │   │
│  │  - QK-Norm      │ states  │  - QK-Norm              │   │
│  │  - RMSNorm      │         │  - Cross-Attention      │   │
│  │  - 28 layers    │         │  - 28 layers            │   │
│  └─────────────────┘         └─────────────────────────┘   │
│           ▲                            │                    │
│           │                            ▼                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Shared Embeddings (152,036 tokens)     │    │
│  │              = Qwen3 vocab + 100 sentinel tokens    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

- Bidirectional encoder (no causal mask)
- Merged self/cross attention in decoder (parameter efficient)
- Tied embeddings across encoder, decoder, and LM head
- 100 sentinel tokens for span corruption training
- Compatible with HuggingFace Transformers

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/qwen2qwen.git
cd qwen2qwen

# Install in editable mode with all dependencies
pip install -e ".[dev,training]"

# Additional dependencies for evaluation
pip install mteb sentence-transformers faiss-cpu scipy scikit-learn
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.40.0
- CUDA GPU with BF16 support (Ampere or newer recommended)
- 24GB+ VRAM for training (inference works on smaller GPUs)

## Quick Start

### 1. Initialize Model from Qwen3

Convert a pretrained Qwen3 model to encoder-decoder format:

```bash
python scripts/initialize_model.py \
    --qwen3-model Qwen/Qwen3-0.6B \
    --output-dir ./qwen3-encdec-initialized \
    --num-sentinel-tokens 100 \
    --verify
```

### 2. Run Sanity Check

Verify the model works correctly:

```bash
python scripts/sanity_check.py \
    --model_path ./qwen3-encdec-initialized \
    --batch_size 2
```

### 3. Train with UL2

Train the model on text data using UL2 denoising objectives:

```bash
# Single GPU training
python scripts/train.py \
    --model_path ./qwen3-encdec-initialized \
    --dataset_name wikimedia/wikipedia \
    --dataset_config 20231101.en \
    --output_dir ./checkpoints \
    --num_train_steps 100000 \
    --batch_size 8 \
    --learning_rate 1e-4

# Multi-GPU with Accelerate
accelerate launch scripts/train.py \
    --model_path ./qwen3-encdec-initialized \
    --dataset_name wikimedia/wikipedia \
    --output_dir ./checkpoints \
    --num_train_steps 100000
```

### 4. Extract Encoder

After training, extract the encoder as a standalone embedding model:

```bash
python scripts/extract_encoder.py \
    --checkpoint_path ./checkpoints/checkpoint-100000 \
    --output_path ./extracted_encoder \
    --export_sentence_transformers \
    --verify
```

### 5. Evaluate

Run evaluation on embedding benchmarks:

```bash
# Quick sanity check
python scripts/quick_eval.py \
    --encoder_path ./extracted_encoder \
    --run_stsb

# Full evaluation
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_sts \
    --run_retrieval \
    --output_dir ./eval_results
```

## Training

### UL2 Denoising Tasks

Training uses the UL2_5 library with `UL25Config.recommended()` for optimized denoising. The default mixture includes:

| Task        | Type      | Description                                        |
| ----------- | --------- | -------------------------------------------------- |
| R-Denoisers | SPAN      | Short/medium span corruption (regular denoising)   |
| X-Denoisers | SPAN      | Long span corruption (extreme denoising)           |
| S-Denoisers | PREFIX    | Prefix-to-suffix generation (sequential denoising) |
| I-Denoiser  | INFILLING | Text infilling (gap filling)                       |

For curriculum learning (task weights shift during training):

```python
from data import ul2_recommended_with_curriculum_config
config = ul2_recommended_with_curriculum_config()
```

### Training Configuration

Create a YAML config file or use command-line arguments:

```yaml
# configs/training_config.yaml
model:
  path: ./qwen3-encdec-initialized

training:
  num_train_steps: 100000
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0

data:
  dataset_name: wikimedia/wikipedia
  dataset_config: 20231101.en
  max_length: 512
  streaming: true

checkpointing:
  save_steps: 5000
  output_dir: ./checkpoints
```

### Distributed Training

For multi-GPU training with FSDP:

```bash
# Configure Accelerate
accelerate config

# Launch training
accelerate launch scripts/train.py --config configs/training_config.yaml
```

### Memory Optimization

The training infrastructure includes several memory optimizations:

- **Gradient Checkpointing**: Enabled by default to reduce memory
- **BF16 Precision**: All training uses BF16 (FP16 is not supported)
- **Streaming Data**: Load data on-the-fly without full materialization
- **Cut Cross Entropy**: Memory-efficient loss computation (24GB → 1MB)
- **Liger Kernels**: Optimized RMSNorm and SwiGLU MLP

## Encoder Extraction

After training, extract the encoder for embedding tasks:

```bash
python scripts/extract_encoder.py \
    --checkpoint_path ./checkpoints/checkpoint-100000 \
    --output_path ./extracted_encoder \
    --pooling_mode mean \
    --export_sentence_transformers \
    --average_checkpoints 3  # Optional: average last N checkpoints
```

### Options

| Flag                             | Description                                             |
| -------------------------------- | ------------------------------------------------------- |
| `--pooling_mode`                 | Pooling strategy: `mean`, `cls`, `last` (default: mean) |
| `--export_sentence_transformers` | Export in sentence-transformers format                  |
| `--average_checkpoints N`        | Average last N checkpoints for robustness               |
| `--verify`                       | Run verification tests after extraction                 |

### Using the Extracted Encoder

```python
from sentence_transformers import SentenceTransformer

# Load the extracted encoder
model = SentenceTransformer('./extracted_encoder')

# Encode sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy canine.",
]
embeddings = model.encode(sentences)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity: {similarity:.4f}")
```

Or use directly with the custom model class:

```python
from qwen3_encdec.encoder_only import Qwen3StandaloneEncoderModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./extracted_encoder')
model = Qwen3StandaloneEncoderModel.from_pretrained('./extracted_encoder')

inputs = tokenizer(["Hello world"], return_tensors="pt", padding=True)
outputs = model(**inputs)
embeddings = outputs.pooler_output  # (batch_size, hidden_size)
```

## Evaluation

### Quick Evaluation

Run fast sanity checks:

```bash
python scripts/quick_eval.py --encoder_path ./extracted_encoder
```

This tests:

- Basic encoding functionality
- Semantic similarity (similar sentences should have higher similarity)
- Embedding diversity (different sentences should produce different embeddings)
- Normalization (embeddings should have unit norm)

### Full Evaluation

#### STS Evaluation (Semantic Textual Similarity)

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_sts \
    --output_dir ./eval_results
```

Evaluates on:

- STS Benchmark (STS-B)
- SICK Relatedness

Metrics: Spearman correlation, Pearson correlation

#### Retrieval Evaluation

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_retrieval \
    --output_dir ./eval_results
```

Metrics: MRR@10, Recall@1/10/100, NDCG@10

#### MTEB Benchmark

Run the full MTEB (Massive Text Embedding Benchmark):

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_mteb \
    --output_dir ./eval_results
```

Or specific tasks:

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_mteb \
    --mteb_tasks STSBenchmark SICKRelatedness \
    --output_dir ./eval_results
```

#### Baseline Comparison

Compare against baseline models:

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --compare_baselines \
    --qwen3_baseline Qwen/Qwen3-0.6B \
    --output_dir ./eval_results
```

Compares against:

- Qwen3-0.6B with mean pooling (baseline without bidirectional training)
- E5-base
- GTE-base
- BGE-base

## Project Structure

```
qwen2qwen/
├── src/
│   ├── qwen3_encdec/           # Core model implementation
│   │   ├── configuration_qwen3_encdec.py
│   │   ├── tokenization_qwen3_encdec.py
│   │   ├── modeling_qwen3_encoder.py
│   │   ├── modeling_qwen3_decoder.py
│   │   ├── modeling_qwen3_encdec.py
│   │   ├── encoder_only.py     # Standalone encoder for inference
│   │   └── weight_initialization.py
│   ├── data/                   # UL2_5-backed data pipeline
│   │   └── ul2_collator.py
│   ├── training/               # Training infrastructure
│   │   ├── config.py
│   │   ├── trainer.py
│   │   ├── execution.py
│   │   ├── monitor.py
│   │   └── memory_utils.py
│   ├── extraction/             # Encoder extraction
│   │   ├── extract_encoder.py
│   │   ├── checkpoint_averaging.py
│   │   └── sentence_transformers_export.py
│   └── evaluation/             # Evaluation utilities
│       ├── mteb_eval.py
│       ├── similarity_eval.py
│       ├── retrieval_eval.py
│       └── baseline_comparison.py
├── scripts/                    # CLI scripts
│   ├── initialize_model.py
│   ├── train.py
│   ├── sanity_check.py
│   ├── validation_run.py
│   ├── extract_encoder.py
│   ├── run_evaluation.py
│   └── quick_eval.py
├── tests/                      # Unit tests
├── configs/                    # Configuration files
└── plan/                       # Implementation stories (01-11)
```

## Testing

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/qwen3_encdec --cov-report=term-missing

# Specific test file
pytest tests/test_evaluation.py -v
```

## Model Details

### Configuration

| Parameter           | Value                             |
| ------------------- | --------------------------------- |
| Hidden Size         | 1024                              |
| Intermediate Size   | 2816                              |
| Num Layers          | 28 (encoder) + 28 (decoder)       |
| Num Attention Heads | 16                                |
| Num KV Heads        | 8 (GQA)                           |
| Vocab Size          | 152,036 (151,936 + 100 sentinels) |
| Max Position        | 32,768                            |
| RoPE Base           | 1,000,000                         |

### Parameter Count

| Component         | Parameters                   |
| ----------------- | ---------------------------- |
| Encoder           | ~300M                        |
| Decoder           | ~300M                        |
| Shared Embeddings | ~156M                        |
| **Total**         | ~600M (with tied embeddings) |

## Troubleshooting

### Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing (enabled by default)
3. Use gradient accumulation
4. Use DeepSpeed ZeRO-3

```bash
python scripts/train.py \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing
```

### Slow Training

1. Enable mixed precision (BF16)
2. Use streaming datasets
3. Increase dataloader workers

### Model Not Learning

1. Check learning rate (try 1e-4 to 5e-5)
2. Verify data pipeline with `--analyze_batch`
3. Check gradient norms in logs

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code, please cite:

```bibtex
@software{qwen3_encdec,
  title = {Qwen3 Encoder-Decoder},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/qwen2qwen}
}
```

## Acknowledgments

- [T5Gemma 2](https://arxiv.org/abs/2XXX.XXXXX) for the architecture pattern
- [Qwen3](https://github.com/QwenLM/Qwen) for the base model
- [UL2](https://arxiv.org/abs/2205.05131) for the training objectives
- [MTEB](https://github.com/embeddings-benchmark/mteb) for evaluation benchmarks
