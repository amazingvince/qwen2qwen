"""Evaluation utilities for Qwen3 Encoder.

This package provides comprehensive evaluation tools for benchmarking
the extracted encoder on embedding tasks.
"""

from .baseline_comparison import BaselineComparison, DecoderPoolWrapper, ModelInfo
from .mteb_eval import MTEBConfig, MTEBEvaluator, Qwen3EncoderWrapper, run_mteb_evaluation
from .retrieval_eval import RetrievalEvaluator, RetrievalResult
from .similarity_eval import STSEvaluator, STSResult

__all__ = [
    # MTEB evaluation
    "MTEBConfig",
    "MTEBEvaluator",
    "Qwen3EncoderWrapper",
    "run_mteb_evaluation",
    # STS evaluation
    "STSEvaluator",
    "STSResult",
    # Retrieval evaluation
    "RetrievalEvaluator",
    "RetrievalResult",
    # Baseline comparison
    "BaselineComparison",
    "DecoderPoolWrapper",
    "ModelInfo",
]
