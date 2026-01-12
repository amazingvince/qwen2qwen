# STS-B Baseline Evaluation Results

Generated: 2026-01-12

## Summary

Baseline STS-B evaluation results for various embedding and encoder models.
All models evaluated on STS-B test set without fine-tuning on STS-B training data.

## Results

| Model | Params | STS-B Spearman | STS-B Pearson |
|-------|--------|----------------|---------------|
| E5-base-v2 | 109M | 0.8548 | 0.8493 |
| Gemma-Embedding-300M | 300M | 0.8487 | 0.8447 |
| Qwen3-Embedding-0.6B | 600M | 0.8460 | 0.8426 |
| all-MiniLM-L6-v2 | 22M | 0.8203 | 0.8274 |
| Ettin-Encoder-1B | 1B | 0.5851 | 0.4565 |
| RoBERTa-base | 125M | 0.5436 | 0.5237 |
| Qwen3-0.6B (zero-train) | 600M | 0.4573 | 0.4183 |
| Ettin-Encoder-400M | 400M | 0.4254 | 0.3560 |
| Ettin-Encoder-17M | 17M | 0.3921 | 0.3262 |

## Notes

- **Qwen3-Embedding-0.6B** uses the same base architecture (Qwen3-0.6B) with contrastive training
- **Qwen3-0.6B (zero-train)** represents the encoder weights at initialization before any training
- Ettin models are encoder-only transformers without contrastive training, evaluated with mean pooling
- RoBERTa-base is evaluated with mean pooling over the final hidden states (no embedding-specific training)
- E5-base-v2, Gemma-Embedding-300M, and all-MiniLM-L6-v2 are sentence-transformers models trained for embeddings
- Spearman correlation (ρ) is the standard metric for STS evaluation

## Evaluation Details

- **Batch size**: 32
- **Device**: cuda
- **Dataset**: STS-B test (1379 sentence pairs)
- **Metric**: Cosine similarity between sentence embeddings
