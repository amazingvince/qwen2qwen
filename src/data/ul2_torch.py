"""
UL2 Corruption Functions - PyTorch Native.

Core span corruption and sentinel processing adapted from UL2_5
(https://github.com/pszemraj/UL2_5 - Apache-2.0 License).

Adapted for Qwen3EncoderDecoderTokenizer sentinel token format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import torch
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================


class Task(IntEnum):
    """Task types as integers for efficient torch operations."""

    SPAN = 0  # Standard T5-style span corruption
    SPAN_MIDDLE = 1  # Position-biased toward middle
    PREFIX_RANDOM = 2  # Random split (20-80%)
    PREFIX_SHORT = 3  # Long prefix, short target
    PREFIX_LONG = 4  # Short prefix, long target
    INFILLING = 5  # Middle-out masking


@dataclass
class DenoiserSpec:
    """Single denoiser specification."""

    task: Task
    mu: float = 3.0  # Mean span length
    r: float = 0.15  # Noise density (corruption ratio)
    max_spans: int = 512
    prefix: str = ""  # Task prefix token (e.g., "[R]", "[X]", "[S]")
    variable_r: bool = False
    r_bounds: Tuple[float, float] = (0.05, 0.50)


@dataclass
class UL2Config:
    """UL2 mixture configuration."""

    denoisers: List[DenoiserSpec] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    @classmethod
    def t5gemma2(cls) -> "UL2Config":
        """
        T5Gemma 2 task configuration with 1:1:1:1:4 weighting.

        Tasks:
        - R1: Short spans (mu=3, r=0.15)
        - R2: Medium spans (mu=12, r=0.50)
        - X1: Long spans, low density (mu=32, r=0.15)
        - X2: Long spans, high density (mu=32, r=0.50)
        - S: Prefix LM (4x weight)
        """
        return cls(
            denoisers=[
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),  # R1
                DenoiserSpec(Task.SPAN, mu=12.0, r=0.50, prefix="[R]"),  # R2
                DenoiserSpec(Task.SPAN, mu=32.0, r=0.15, prefix="[X]"),  # X1
                DenoiserSpec(Task.SPAN, mu=32.0, r=0.50, prefix="[X]"),  # X2
                DenoiserSpec(Task.PREFIX_RANDOM, r=0.75, prefix="[S]"),  # S
            ],
            weights=[1, 1, 1, 1, 4],  # S-denoiser gets 4x weight
        )

    @classmethod
    def recommended(cls) -> "UL2Config":
        """Recommended mixture based on UL2.5 feasibility analysis."""
        return cls(
            denoisers=[
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN_MIDDLE, mu=16.0, r=0.20, prefix="[X]"),
                DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(Task.PREFIX_SHORT, prefix="[S]"),
                DenoiserSpec(Task.PREFIX_LONG, prefix="[S]"),
                DenoiserSpec(Task.INFILLING, r=0.30, prefix="[I]"),
            ],
            weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
        )

    @classmethod
    def span_heavy(cls) -> "UL2Config":
        """Original UL2-style with more span denoising."""
        return cls(
            denoisers=[
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.50, prefix="[X]"),
                DenoiserSpec(Task.SPAN, mu=64.0, r=0.50, prefix="[X]"),
                DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
            ],
            weights=[0.20, 0.20, 0.15, 0.15, 0.30],
        )

    @classmethod
    def minimal(cls) -> "UL2Config":
        """Minimal config for testing."""
        return cls(
            denoisers=[
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15),
                DenoiserSpec(Task.PREFIX_RANDOM),
            ],
            weights=[0.5, 0.5],
        )


# =============================================================================
# MASKING FUNCTIONS (Vectorized PyTorch)
# =============================================================================


def _random_segmentation(
    num_items: int,
    num_segments: int,
    device: torch.device,
) -> Tensor:
    """Partition num_items into num_segments non-empty segments."""
    if num_segments <= 0 or num_items <= 0:
        return torch.ones(1, dtype=torch.long, device=device)
    if num_segments >= num_items:
        return torch.ones(num_items, dtype=torch.long, device=device)

    # Sample divider positions
    dividers = torch.randperm(num_items - 1, device=device)[: num_segments - 1]
    dividers = torch.sort(dividers).values

    # Compute segment lengths from divider positions
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            dividers + 1,
        ]
    )
    ends = torch.cat(
        [
            dividers + 1,
            torch.tensor([num_items], dtype=torch.long, device=device),
        ]
    )

    return ends - starts


def span_corruption_mask(
    seq_len: int,
    noise_density: float,
    mean_span_length: float,
    max_spans: int,
    device: torch.device,
) -> Tensor:
    """
    Generate T5-style span corruption mask.

    Args:
        seq_len: Length of sequence to mask.
        noise_density: Target fraction of tokens to corrupt.
        mean_span_length: Mean length of each corruption span.
        max_spans: Maximum number of spans.
        device: Torch device.

    Returns:
        Boolean tensor [seq_len] where True = corrupted/masked.
    """
    num_noise = max(1, min(int(round(seq_len * noise_density)), seq_len - 1))
    num_spans = max(1, min(max_spans, int(round(num_noise / mean_span_length))))
    num_nonnoise = seq_len - num_noise

    noise_lengths = _random_segmentation(num_noise, num_spans, device)
    nonnoise_lengths = _random_segmentation(num_nonnoise, num_spans, device)

    # Interleave segments with random start to avoid edge bias
    n = min(len(noise_lengths), len(nonnoise_lengths))
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    pos = 0

    # Convert to lists once to avoid repeated .item() calls
    noise_list = noise_lengths.tolist()
    nonnoise_list = nonnoise_lengths.tolist()

    # Randomize starting pattern to eliminate edge effect
    start_with_noise = torch.rand(1, device=device).item() < 0.5

    for i in range(n):
        if start_with_noise:
            # Noise segment first
            noise_len = noise_list[i]
            end = min(pos + noise_len, seq_len)
            mask[pos:end] = True
            pos = end
            # Then nonnoise segment
            pos += nonnoise_list[i]
        else:
            # Nonnoise segment first
            pos += nonnoise_list[i]
            # Then noise segment
            noise_len = noise_list[i]
            end = min(pos + noise_len, seq_len)
            mask[pos:end] = True
            pos = end
        if pos >= seq_len:
            break

    return mask


def middle_heavy_mask(
    seq_len: int,
    noise_density: float,
    device: torch.device,
) -> Tensor:
    """
    Position-biased mask preferring middle positions.

    Uses Gaussian weighting centered at sequence middle.
    """
    num_noise = max(1, min(int(round(seq_len * noise_density)), seq_len - 1))

    # Gaussian weights
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    center = seq_len / 2
    sigma = seq_len / 4
    weights = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
    weights = weights / weights.sum()

    # Sample without replacement
    indices = torch.multinomial(weights, num_noise, replacement=False)

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask


def prefix_lm_mask(
    seq_len: int,
    mode: str,  # "random", "short", "long"
    device: torch.device,
) -> Tuple[Tensor, int]:
    """
    Generate prefix LM mask.

    Args:
        seq_len: Sequence length.
        mode: "random" (20-80%), "short" (5-15% target), "long" (5-20% prefix).
        device: Torch device.

    Returns:
        (mask, split_index) where mask[split:] = True.
    """
    if mode == "random":
        min_s, max_s = int(0.2 * seq_len), int(0.8 * seq_len)
        split = torch.randint(min_s, max_s + 1, (1,), device=device).item()
    elif mode == "short":
        # Short target: 5-15% of sequence
        frac = 0.05 + 0.10 * torch.rand(1, device=device).item()
        split = int((1 - frac) * seq_len)
    elif mode == "long":
        # Long target: prefix is 5-20%
        frac = 0.05 + 0.15 * torch.rand(1, device=device).item()
        split = int(frac * seq_len)
    else:
        split = int(0.75 * seq_len)

    split = max(1, min(split, seq_len - 1))

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[split:] = True
    return mask, split


def infilling_mask(
    seq_len: int,
    hole_frac: float,
    device: torch.device,
) -> Tuple[Tensor, int, int]:
    """
    Generate infilling mask (mask middle portion).

    Returns:
        (mask, hole_start, hole_end).
    """
    hole_size = max(1, int(hole_frac * seq_len))

    min_start = int(0.1 * seq_len)
    max_start = max(min_start, int(0.9 * seq_len) - hole_size)

    if max_start <= min_start:
        hole_start = seq_len // 3
    else:
        hole_start = torch.randint(min_start, max_start + 1, (1,), device=device).item()

    hole_end = min(hole_start + hole_size, seq_len)

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[hole_start:hole_end] = True
    return mask, hole_start, hole_end


# =============================================================================
# SENTINEL PROCESSING
# =============================================================================


def create_sentinel_ids(
    mask: Tensor,
    sentinel_start_id: int,
) -> Tensor:
    """
    Convert boolean mask to sentinel token IDs.

    Adapted for Qwen3EncoderDecoderTokenizer where sentinel IDs
    count UP from sentinel_start_id (original_vocab_size).

    Args:
        mask: Boolean tensor [seq_len] where True = masked position.
        sentinel_start_id: First sentinel token ID (e.g., 151936 for <extra_id_0>).

    Returns:
        Tensor where:
        - Span starts have sentinel IDs (sentinel_start_id, sentinel_start_id+1, ...)
        - Continuation positions have -1 (to be removed)
        - Unmasked positions have 0
    """
    device = mask.device

    # Find span starts: transition from False->True
    shifted = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]])
    span_starts = mask & ~shifted

    # Assign increasing sentinel IDs to span starts
    # First span gets sentinel_start_id (<extra_id_0>)
    # Second span gets sentinel_start_id + 1 (<extra_id_1>), etc.
    cumsum = torch.cumsum(span_starts.int(), dim=0)

    sentinel_ids = torch.where(
        span_starts,
        sentinel_start_id + cumsum - 1,  # -1 because cumsum starts at 1
        torch.zeros_like(cumsum),
    )

    # Mark continuation positions with -1
    continuations = mask & ~span_starts
    sentinel_ids = torch.where(
        continuations,
        torch.full_like(sentinel_ids, -1),
        sentinel_ids,
    )

    return sentinel_ids


def apply_sentinel_mask(
    input_ids: Tensor,
    sentinel_ids: Tensor,
    prefix_ids: Optional[Tensor] = None,
    eos_id: Optional[int] = None,
) -> Tensor:
    """
    Apply sentinel mask: replace spans with sentinels, remove continuations.

    Args:
        input_ids: Original token IDs [seq_len].
        sentinel_ids: From create_sentinel_ids [seq_len].
        prefix_ids: Optional prefix tokens to prepend.
        eos_id: Optional EOS token to append.

    Returns:
        Filtered token IDs with sentinels.
    """
    device = input_ids.device

    # Replace masked positions with sentinels
    result = torch.where(sentinel_ids > 0, sentinel_ids, input_ids)

    # Filter out -1 positions (continuations)
    keep_mask = sentinel_ids != -1
    result = result[keep_mask]

    # Prepend prefix
    if prefix_ids is not None and prefix_ids.numel() > 0:
        result = torch.cat([prefix_ids.to(device), result])

    # Append EOS
    if eos_id is not None:
        result = torch.cat(
            [result, torch.tensor([eos_id], dtype=result.dtype, device=device)]
        )

    return result


def count_num_spans(mask: Tensor) -> int:
    """Count the number of contiguous spans in a mask."""
    device = mask.device
    shifted = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]])
    span_starts = mask & ~shifted
    return span_starts.sum().item()
