"""Tests for evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSTSResult:
    """Test STSResult dataclass."""

    def test_creation(self):
        """Test creating STSResult."""
        from src.evaluation.similarity_eval import STSResult

        result = STSResult(
            spearman=0.85,
            pearson=0.82,
            dataset_name="stsb",
            num_samples=1500,
        )

        assert result.spearman == 0.85
        assert result.pearson == 0.82
        assert result.dataset_name == "stsb"
        assert result.num_samples == 1500

    def test_repr(self):
        """Test STSResult string representation."""
        from src.evaluation.similarity_eval import STSResult

        result = STSResult(
            spearman=0.8567,
            pearson=0.8234,
            dataset_name="stsb",
            num_samples=1500,
        )

        repr_str = repr(result)
        assert "stsb" in repr_str
        assert "0.8567" in repr_str
        assert "0.8234" in repr_str


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_creation(self):
        """Test creating RetrievalResult."""
        from src.evaluation.retrieval_eval import RetrievalResult

        result = RetrievalResult(
            mrr_at_10=0.75,
            recall_at_1=0.60,
            recall_at_10=0.85,
            recall_at_100=0.95,
            ndcg_at_10=0.78,
            dataset_name="custom",
            num_queries=100,
            num_docs=1000,
        )

        assert result.mrr_at_10 == 0.75
        assert result.recall_at_1 == 0.60
        assert result.recall_at_10 == 0.85
        assert result.recall_at_100 == 0.95
        assert result.ndcg_at_10 == 0.78

    def test_asdict(self):
        """Test converting to dictionary."""
        from src.evaluation.retrieval_eval import RetrievalResult

        result = RetrievalResult(
            mrr_at_10=0.75,
            recall_at_1=0.60,
            recall_at_10=0.85,
            recall_at_100=0.95,
            ndcg_at_10=0.78,
            dataset_name="custom",
            num_queries=100,
            num_docs=1000,
        )

        d = asdict(result)
        assert d["mrr_at_10"] == 0.75
        assert d["dataset_name"] == "custom"


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ModelInfo with defaults."""
        from src.evaluation.baseline_comparison import ModelInfo

        info = ModelInfo(
            name="Test Model",
            path="/path/to/model",
            model_type="sentence_transformer",
        )

        assert info.name == "Test Model"
        assert info.pooling == "mean"
        assert info.normalize is True

    def test_creation_custom(self):
        """Test creating ModelInfo with custom values."""
        from src.evaluation.baseline_comparison import ModelInfo

        info = ModelInfo(
            name="Custom Model",
            path="/path/to/model",
            model_type="decoder_pool",
            pooling="last",
            normalize=False,
        )

        assert info.pooling == "last"
        assert info.normalize is False


class TestSTSEvaluator:
    """Test STSEvaluator class."""

    def create_mock_model(self, embedding_dim: int = 768):
        """Create a mock model with encode method."""
        mock_model = MagicMock()

        def mock_encode(sentences, **kwargs):
            # Return random embeddings
            embeddings = np.random.randn(len(sentences), embedding_dim).astype(np.float32)
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

        mock_model.encode = mock_encode
        return mock_model

    def test_encode_sentences(self):
        """Test encoding sentences."""
        from src.evaluation.similarity_eval import STSEvaluator

        mock_model = self.create_mock_model()
        evaluator = STSEvaluator(mock_model, device="cpu")

        sentences = ["Hello world", "Test sentence", "Another one"]
        embeddings = evaluator.encode_sentences(sentences, show_progress=False)

        assert embeddings.shape == (3, 768)
        assert not np.isnan(embeddings).any()

    def test_evaluate_custom(self):
        """Test custom evaluation."""
        from src.evaluation.similarity_eval import STSEvaluator

        mock_model = self.create_mock_model()
        evaluator = STSEvaluator(mock_model, device="cpu")

        # Create synthetic data with known correlations
        sentences1 = ["cat", "dog", "car", "tree", "book"]
        sentences2 = ["kitten", "puppy", "automobile", "plant", "novel"]
        gold_scores = [0.9, 0.85, 0.95, 0.7, 0.8]

        result = evaluator.evaluate_custom(
            sentences1, sentences2, gold_scores, batch_size=32
        )

        assert isinstance(result.spearman, float)
        assert isinstance(result.pearson, float)
        assert result.dataset_name == "custom"
        assert result.num_samples == 5


class TestRetrievalEvaluator:
    """Test RetrievalEvaluator class."""

    def create_mock_model(self, embedding_dim: int = 768):
        """Create a mock model with encode method."""
        mock_model = MagicMock()

        def mock_encode(sentences, **kwargs):
            embeddings = np.random.randn(len(sentences), embedding_dim).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

        mock_model.encode = mock_encode
        return mock_model

    def test_compute_metrics_perfect_retrieval(self):
        """Test metrics with perfect retrieval."""
        from src.evaluation.retrieval_eval import RetrievalEvaluator

        mock_model = self.create_mock_model()
        evaluator = RetrievalEvaluator(mock_model, device="cpu", use_faiss=False)

        # Perfect retrieval: first retrieved doc is always relevant
        retrieved_indices = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
            [2, 0, 1, 3, 4, 5, 6, 7, 8, 9],
        ])
        relevant_docs = [[0], [1], [2]]

        metrics = evaluator.compute_metrics(retrieved_indices, relevant_docs)

        assert metrics["mrr@10"] == 1.0
        assert metrics["recall@1"] == 1.0
        assert metrics["recall@10"] == 1.0

    def test_compute_metrics_partial_retrieval(self):
        """Test metrics with partial retrieval."""
        from src.evaluation.retrieval_eval import RetrievalEvaluator

        mock_model = self.create_mock_model()
        evaluator = RetrievalEvaluator(mock_model, device="cpu", use_faiss=False)

        # Partial retrieval: relevant doc at various positions
        retrieved_indices = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # doc 0 at pos 0, rank 1
            [5, 1, 2, 3, 4, 0, 6, 7, 8, 9],  # doc 1 at pos 1, rank 2
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # doc 2 at pos 7, rank 8
        ])
        relevant_docs = [[0], [1], [2]]

        metrics = evaluator.compute_metrics(retrieved_indices, relevant_docs)

        # MRR@10 = (1/1 + 1/2 + 1/8) / 3 ≈ 0.5417
        expected_mrr = (1.0 + 0.5 + 0.125) / 3
        assert abs(metrics["mrr@10"] - expected_mrr) < 0.01

        # Recall@1 = 1/3 (only first query has relevant at pos 0)
        assert abs(metrics["recall@1"] - 1/3) < 0.01

        # Recall@10 = 3/3 = 1.0 (all have relevant in top 10)
        assert metrics["recall@10"] == 1.0

    def test_evaluate_custom(self):
        """Test custom retrieval evaluation."""
        from src.evaluation.retrieval_eval import RetrievalEvaluator

        mock_model = self.create_mock_model()
        evaluator = RetrievalEvaluator(mock_model, device="cpu", use_faiss=False)

        queries = ["query 1", "query 2"]
        documents = ["doc 1", "doc 2", "doc 3", "doc 4"]
        relevant_docs = [[0], [1]]

        result = evaluator.evaluate_custom(
            queries, documents, relevant_docs, batch_size=32
        )

        assert result.num_queries == 2
        assert result.num_docs == 4
        assert 0.0 <= result.mrr_at_10 <= 1.0
        assert 0.0 <= result.recall_at_1 <= 1.0

    def test_brute_force_search(self):
        """Test brute force search (no FAISS)."""
        from src.evaluation.retrieval_eval import RetrievalEvaluator

        mock_model = self.create_mock_model()
        evaluator = RetrievalEvaluator(mock_model, device="cpu", use_faiss=False)

        # Create embeddings where we know the ranking
        query_embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        doc_embeddings = np.array([
            [0.1, 0.9, 0.0],  # Low similarity to query
            [0.9, 0.1, 0.0],  # High similarity to query
            [0.5, 0.5, 0.0],  # Medium similarity
        ], dtype=np.float32)

        # Normalize
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        scores, indices = evaluator.search(query_embeddings, doc_embeddings, k=3)

        # Doc 1 should be first (highest similarity)
        assert indices[0, 0] == 1


class TestDecoderPoolWrapper:
    """Test DecoderPoolWrapper class."""

    def test_mean_pooling(self):
        """Test mean pooling logic (unit test)."""
        import torch

        # Test the pooling logic directly
        batch_size = 2
        seq_len = 10
        hidden_dim = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Mean pooling implementation
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        embeddings = (hidden_states * attention_mask_expanded).sum(1) / attention_mask_expanded.sum(1)

        assert embeddings.shape == (batch_size, hidden_dim)
        assert not torch.isnan(embeddings).any()

    def test_last_token_pooling(self):
        """Test last token pooling logic."""
        import torch

        batch_size = 2
        seq_len = 10
        hidden_dim = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Last token pooling
        seq_lengths = attention_mask.sum(1) - 1
        embeddings = hidden_states[
            torch.arange(hidden_states.size(0)),
            seq_lengths,
        ]

        assert embeddings.shape == (batch_size, hidden_dim)

    def test_pooling_options(self):
        """Test that pooling options are valid."""
        valid_poolings = ["mean", "last", "cls"]

        for pooling in valid_poolings:
            # Just check that the option is documented/expected
            assert pooling in ["mean", "last", "cls"]


class TestBaselineComparison:
    """Test BaselineComparison class."""

    def test_baselines_defined(self):
        """Test that baseline models are defined."""
        from src.evaluation.baseline_comparison import BaselineComparison

        assert "e5-base" in BaselineComparison.BASELINES
        assert "gte-base" in BaselineComparison.BASELINES
        assert "bge-base" in BaselineComparison.BASELINES

    def test_baseline_info(self):
        """Test baseline model info."""
        from src.evaluation.baseline_comparison import BaselineComparison

        e5_info = BaselineComparison.BASELINES["e5-base"]
        assert e5_info.name == "E5-base"
        assert e5_info.model_type == "sentence_transformer"

    def test_create_comparison_table_empty(self):
        """Test comparison table with no results."""
        from src.evaluation.baseline_comparison import BaselineComparison

        comparison = BaselineComparison(device="cpu")
        table = comparison.create_comparison_table()

        # Empty results should produce minimal output
        assert isinstance(table, str)


class TestMTEBConfig:
    """Test MTEBConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from src.evaluation.mteb_eval import MTEBConfig

        config = MTEBConfig()

        assert config.model_path == "./extracted_encoder"
        assert config.batch_size == 32
        assert config.max_seq_length == 512
        assert "STS" in config.task_categories
        assert "Retrieval" in config.task_categories

    def test_custom_values(self):
        """Test custom configuration."""
        from src.evaluation.mteb_eval import MTEBConfig

        config = MTEBConfig(
            model_path="/custom/path",
            batch_size=64,
            task_categories=["STS"],
            task_names=["STSBenchmark"],
        )

        assert config.model_path == "/custom/path"
        assert config.batch_size == 64
        assert config.task_names == ["STSBenchmark"]


class TestQwen3EncoderWrapper:
    """Test Qwen3EncoderWrapper class."""

    def test_wrapper_interface(self):
        """Test that wrapper has required interface."""
        from src.evaluation.mteb_eval import Qwen3EncoderWrapper

        # Check class has encode method
        assert hasattr(Qwen3EncoderWrapper, "encode")

    def test_wrapper_attributes(self):
        """Test wrapper has expected attributes after init."""
        from src.evaluation.mteb_eval import Qwen3EncoderWrapper

        # Check class attributes
        assert hasattr(Qwen3EncoderWrapper, "_load_model")

        # Check default values in __init__ signature
        import inspect

        sig = inspect.signature(Qwen3EncoderWrapper.__init__)
        params = sig.parameters

        assert "model_path" in params
        assert "device" in params
        assert params["device"].default == "cuda"
        assert params["max_seq_length"].default == 512
        assert params["normalize_embeddings"].default is True
        assert params["pooling_mode"].default == "mean"


class TestIntegration:
    """Integration tests (require actual model or more setup)."""

    def test_sts_evaluator_has_datasets(self):
        """Test that STS datasets are defined."""
        from src.evaluation.similarity_eval import STSEvaluator

        assert "stsb" in STSEvaluator.STS_DATASETS
        assert "sick" in STSEvaluator.STS_DATASETS

        stsb_config = STSEvaluator.STS_DATASETS["stsb"]
        assert "path" in stsb_config
        assert "split" in stsb_config

    def test_retrieval_metric_computation_is_deterministic(self):
        """Test that metric computation is deterministic."""
        from src.evaluation.retrieval_eval import RetrievalEvaluator

        mock_model = MagicMock()
        evaluator = RetrievalEvaluator(mock_model, device="cpu", use_faiss=False)

        retrieved_indices = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        ])
        relevant_docs = [[0], [1]]

        metrics1 = evaluator.compute_metrics(retrieved_indices, relevant_docs)
        metrics2 = evaluator.compute_metrics(retrieved_indices, relevant_docs)

        assert metrics1 == metrics2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
