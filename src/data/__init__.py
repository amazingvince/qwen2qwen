"""
UL2 Data Pipeline for Qwen3 Encoder-Decoder.

Provides span corruption and collation utilities adapted from UL2_5
(https://github.com/pszemraj/UL2_5 - Apache-2.0 License).
"""

from .ul2_collator import UL2DataCollator
from .ul2_torch import (
    DenoiserSpec,
    Task,
    UL2Config,
    apply_sentinel_mask,
    count_num_spans,
    create_sentinel_ids,
    infilling_mask,
    middle_heavy_mask,
    prefix_lm_mask,
    span_corruption_mask,
)

__all__ = [
    # Collator
    "UL2DataCollator",
    # Configuration
    "UL2Config",
    "DenoiserSpec",
    "Task",
    # Masking functions
    "span_corruption_mask",
    "middle_heavy_mask",
    "prefix_lm_mask",
    "infilling_mask",
    # Sentinel processing
    "create_sentinel_ids",
    "apply_sentinel_mask",
    "count_num_spans",
]
