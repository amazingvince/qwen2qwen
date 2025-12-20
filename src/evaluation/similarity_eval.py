"""Semantic Textual Similarity (STS) evaluation.

Evaluates embedding quality on sentence similarity tasks:
- STS Benchmark (STS-B)
- SICK Relatedness
- Custom similarity datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


@dataclass
class STSResult:
    """Results from STS evaluation."""

    spearman: float
    pearson: float
    dataset_name: str
    num_samples: int

    def __repr__(self):
        return (
            f"STSResult(dataset={self.dataset_name}, "
            f"spearman={self.spearman:.4f}, "
            f"pearson={self.pearson:.4f}, "
            f"n={self.num_samples})"
        )


class STSEvaluator:
    """Evaluator for Semantic Textual Similarity tasks."""

    # Standard STS datasets
    STS_DATASETS = {
        "stsb": {
            "path": "mteb/stsbenchmark-sts",
            "split": "test",
            "sent1_col": "sentence1",
            "sent2_col": "sentence2",
            "score_col": "score",
            "score_range": (0, 5),
        },
        "sick": {
            "path": "sick",
            "split": "test",
            "sent1_col": "sentence_A",
            "sent2_col": "sentence_B",
            "score_col": "relatedness_score",
            "score_range": (1, 5),
        },
    }

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        device: str = "cuda",
    ):
        """
        Initialize STS evaluator.

        Args:
            model: Encoder model or model wrapper with encode() method
            tokenizer: Tokenizer (optional if model has encode method)
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else "cpu"

        # Check if model has encode method
        self.has_encode = hasattr(model, "encode")

    def encode_sentences(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode sentences to embeddings."""
        if self.has_encode:
            return self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

        # Manual encoding
        all_embeddings = []

        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for start_idx in iterator:
                batch = sentences[start_idx : start_idx + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()
                all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def evaluate_dataset(
        self,
        dataset_name: str,
        batch_size: int = 32,
    ) -> STSResult:
        """
        Evaluate on a standard STS dataset.

        Args:
            dataset_name: Name of dataset (stsb, sick)
            batch_size: Batch size for encoding

        Returns:
            STSResult with correlation metrics
        """
        if dataset_name not in self.STS_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required. Install with: pip install datasets"
            )

        config = self.STS_DATASETS[dataset_name]

        # Load dataset
        dataset = load_dataset(config["path"], split=config["split"])

        sentences1 = dataset[config["sent1_col"]]
        sentences2 = dataset[config["sent2_col"]]
        gold_scores = np.array(dataset[config["score_col"]])

        # Normalize scores to [0, 1]
        min_score, max_score = config["score_range"]
        gold_scores = (gold_scores - min_score) / (max_score - min_score)

        # Encode sentences
        embeddings1 = self.encode_sentences(sentences1, batch_size)
        embeddings2 = self.encode_sentences(sentences2, batch_size)

        # Compute cosine similarities
        pred_scores = np.array(
            [
                cosine_similarity([e1], [e2])[0, 0]
                for e1, e2 in zip(embeddings1, embeddings2)
            ]
        )

        # Compute correlations
        spearman = spearmanr(gold_scores, pred_scores)[0]
        pearson = pearsonr(gold_scores, pred_scores)[0]

        return STSResult(
            spearman=float(spearman),
            pearson=float(pearson),
            dataset_name=dataset_name,
            num_samples=len(gold_scores),
        )

    def evaluate_all(self, batch_size: int = 32) -> Dict[str, STSResult]:
        """Evaluate on all standard STS datasets."""
        results = {}

        for dataset_name in self.STS_DATASETS:
            print(f"\nEvaluating on {dataset_name}...")
            results[dataset_name] = self.evaluate_dataset(dataset_name, batch_size)
            print(results[dataset_name])

        return results

    def evaluate_custom(
        self,
        sentences1: List[str],
        sentences2: List[str],
        gold_scores: List[float],
        batch_size: int = 32,
    ) -> STSResult:
        """
        Evaluate on custom sentence pairs.

        Args:
            sentences1: First sentences
            sentences2: Second sentences
            gold_scores: Gold similarity scores (0-1)
            batch_size: Batch size for encoding

        Returns:
            STSResult with correlation metrics
        """
        embeddings1 = self.encode_sentences(sentences1, batch_size)
        embeddings2 = self.encode_sentences(sentences2, batch_size)

        pred_scores = np.array(
            [
                cosine_similarity([e1], [e2])[0, 0]
                for e1, e2 in zip(embeddings1, embeddings2)
            ]
        )

        gold_scores_arr = np.array(gold_scores)

        spearman = spearmanr(gold_scores_arr, pred_scores)[0]
        pearson = pearsonr(gold_scores_arr, pred_scores)[0]

        return STSResult(
            spearman=float(spearman),
            pearson=float(pearson),
            dataset_name="custom",
            num_samples=len(gold_scores),
        )
