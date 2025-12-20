"""Retrieval evaluation for embedding models.

Evaluates on:
- MS MARCO passage retrieval
- Natural Questions
- Custom retrieval datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class RetrievalResult:
    """Results from retrieval evaluation."""

    mrr_at_10: float
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    dataset_name: str
    num_queries: int
    num_docs: int

    def __repr__(self):
        return (
            f"RetrievalResult(dataset={self.dataset_name}, "
            f"MRR@10={self.mrr_at_10:.4f}, "
            f"R@1={self.recall_at_1:.4f}, "
            f"R@10={self.recall_at_10:.4f}, "
            f"NDCG@10={self.ndcg_at_10:.4f})"
        )


class RetrievalEvaluator:
    """Evaluator for retrieval tasks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        device: str = "cuda",
        use_faiss: bool = True,
    ):
        """
        Initialize retrieval evaluator.

        Args:
            model: Encoder model or wrapper with encode() method
            tokenizer: Tokenizer (optional if model has encode)
            device: Device for computation
            use_faiss: Whether to use FAISS for efficient similarity search
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_faiss = use_faiss
        self.has_encode = hasattr(model, "encode")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        if prefix:
            texts = [prefix + t for t in texts]

        if self.has_encode:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        # Manual encoding with normalization
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for start_idx in iterator:
                batch = texts[start_idx : start_idx + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def build_index(self, embeddings: np.ndarray) -> Any:
        """Build FAISS index for efficient similarity search."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for efficient search. "
                "Install with: pip install faiss-cpu"
            )

        dim = embeddings.shape[1]

        # Use inner product (equivalent to cosine for normalized vectors)
        index = faiss.IndexFlatIP(dim)

        # Add embeddings
        embeddings = embeddings.astype(np.float32)
        index.add(embeddings)

        return index

    def search(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k documents for each query.

        Args:
            query_embeddings: Query embeddings (num_queries, dim)
            doc_embeddings: Document embeddings (num_docs, dim)
            k: Number of results per query

        Returns:
            scores: (num_queries, k) similarity scores
            indices: (num_queries, k) document indices
        """
        if self.use_faiss:
            try:
                index = self.build_index(doc_embeddings)
                scores, indices = index.search(query_embeddings.astype(np.float32), k)
            except ImportError:
                # Fall back to brute force
                self.use_faiss = False
                return self.search(query_embeddings, doc_embeddings, k)
        else:
            # Brute force search
            similarities = query_embeddings @ doc_embeddings.T
            indices = np.argsort(-similarities, axis=1)[:, :k]
            scores = np.take_along_axis(similarities, indices, axis=1)

        return scores, indices

    def compute_metrics(
        self,
        retrieved_indices: np.ndarray,
        relevant_docs: List[List[int]],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.

        Args:
            retrieved_indices: (num_queries, max_k) retrieved document indices
            relevant_docs: List of relevant doc indices for each query
            k_values: K values for recall@k

        Returns:
            Dictionary of metrics
        """
        if k_values is None:
            k_values = [1, 10, 100]

        num_queries = len(relevant_docs)

        # MRR@10
        mrr = 0.0
        for i, rel in enumerate(relevant_docs):
            rel_set = set(rel)
            for rank, doc_idx in enumerate(retrieved_indices[i, :10]):
                if doc_idx in rel_set:
                    mrr += 1.0 / (rank + 1)
                    break
        mrr /= num_queries

        # Recall@k
        recalls = {}
        for k in k_values:
            recall = 0.0
            for i, rel in enumerate(relevant_docs):
                rel_set = set(rel)
                retrieved_set = set(retrieved_indices[i, :k].tolist())
                recall += len(rel_set & retrieved_set) / max(len(rel_set), 1)
            recalls[f"recall@{k}"] = recall / num_queries

        # NDCG@10
        ndcg = 0.0
        for i, rel in enumerate(relevant_docs):
            rel_set = set(rel)
            dcg = 0.0
            for rank, doc_idx in enumerate(retrieved_indices[i, :10]):
                if doc_idx in rel_set:
                    dcg += 1.0 / np.log2(rank + 2)

            # Ideal DCG
            ideal_dcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), 10)))
            ndcg += dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg /= num_queries

        return {
            "mrr@10": mrr,
            "ndcg@10": ndcg,
            **recalls,
        }

    def evaluate_custom(
        self,
        queries: List[str],
        documents: List[str],
        relevant_docs: List[List[int]],
        batch_size: int = 32,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> RetrievalResult:
        """
        Evaluate on custom retrieval dataset.

        Args:
            queries: Query texts
            documents: Document texts
            relevant_docs: List of relevant doc indices per query
            batch_size: Batch size for encoding
            query_prefix: Prefix for queries
            doc_prefix: Prefix for documents

        Returns:
            RetrievalResult with metrics
        """
        # Encode
        doc_embeddings = self.encode_texts(documents, batch_size, prefix=doc_prefix)
        query_embeddings = self.encode_texts(queries, batch_size, prefix=query_prefix)

        # Search
        scores, indices = self.search(query_embeddings, doc_embeddings, k=100)

        # Compute metrics
        metrics = self.compute_metrics(indices, relevant_docs)

        return RetrievalResult(
            mrr_at_10=metrics["mrr@10"],
            recall_at_1=metrics["recall@1"],
            recall_at_10=metrics["recall@10"],
            recall_at_100=metrics["recall@100"],
            ndcg_at_10=metrics["ndcg@10"],
            dataset_name="custom",
            num_queries=len(queries),
            num_docs=len(documents),
        )
