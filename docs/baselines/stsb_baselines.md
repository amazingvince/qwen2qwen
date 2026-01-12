# STS-B Baseline Evaluation Results

Generated: 2026-01-12

## Summary

These baselines establish reference values for monitoring Qwen3-Encoder training progress.
Models are evaluated on STS-B test set without fine-tuning on STS-B training data.

## Results

| Model | Params | STS-B Spearman | STS-B Pearson |
|-------|--------|----------------|---------------|
| E5-base-v2 | 109M | 0.8548 | 0.8493 |
| Qwen3-Embedding-0.6B | 600M | 0.8460 | 0.8426 |
| all-MiniLM-L6-v2 | 22M | 0.8203 | 0.8274 |
| Ettin-Encoder-1B | 1B | 0.5851 | 0.4565 |
| RoBERTa-base | 125M | 0.5436 | 0.5237 |
| Ettin-Encoder-400M | 400M | 0.4254 | 0.3560 |

## Notes

- **Qwen3-Embedding-0.6B** is the primary target for our trained encoder to match/beat
- Ettin models are encoder-only transformers (no contrastive training), evaluated with mean pooling
- RoBERTa-base is a raw encoder baseline with mean pooling (no embedding training)
- E5-base-v2 and all-MiniLM-L6-v2 are sentence-transformers models for reference
- Spearman correlation (rho) is the primary metric for STS tasks

## Evaluation Details

- **Batch size**: 32
- **Device**: cuda
- **Dataset**: STS-B test (1379 sentence pairs)
- **Metric**: Cosine similarity between sentence embeddings
