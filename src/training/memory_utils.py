"""Memory optimization utilities for training."""

import gc
import logging
from typing import Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def get_memory_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics.

    Returns:
        Dictionary with memory stats in GB, or {"gpu_available": False} if no GPU.
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    return {
        "gpu_available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def log_memory_stats(prefix: str = "") -> None:
    """
    Log current memory statistics.

    Args:
        prefix: Optional prefix for log message.
    """
    stats = get_memory_stats()
    if stats.get("gpu_available", False):
        logger.info(
            f"{prefix}Memory: allocated={stats['allocated_gb']:.2f}GB, "
            f"reserved={stats['reserved_gb']:.2f}GB, "
            f"max={stats['max_allocated_gb']:.2f}GB"
        )


def clear_memory() -> None:
    """Clear GPU memory cache and reset peak memory stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def estimate_model_memory(
    num_params: int,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    dtype: torch.dtype = torch.bfloat16,
    optimizer: str = "adamw",
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.

    Args:
        num_params: Number of model parameters.
        batch_size: Training batch size.
        seq_length: Sequence length.
        hidden_size: Model hidden dimension.
        num_layers: Number of transformer layers.
        dtype: Model dtype (bfloat16 or float32).
        optimizer: Optimizer type ("adamw" or "sgd").
        gradient_checkpointing: Whether gradient checkpointing is enabled.

    Returns:
        Dictionary with memory estimates in GB.
    """
    bytes_per_param = 2 if dtype == torch.bfloat16 else 4

    # Model parameters
    model_memory = num_params * bytes_per_param / 1e9

    # Optimizer states (AdamW: 2 FP32 moments per param)
    if optimizer == "adamw":
        optimizer_memory = num_params * 4 * 2 / 1e9  # 2 FP32 states
    elif optimizer == "sgd":
        optimizer_memory = num_params * 4 / 1e9  # 1 FP32 momentum
    else:
        optimizer_memory = 0

    # Gradients
    gradient_memory = num_params * bytes_per_param / 1e9

    # Activations (rough estimate)
    # Each layer stores activations of size: batch * seq * hidden * 2 (forward + backward)
    if gradient_checkpointing:
        # With checkpointing, only sqrt(layers) activations stored
        activation_layers = int(num_layers**0.5) + 1
    else:
        activation_layers = num_layers

    activation_memory = (
        batch_size
        * seq_length
        * hidden_size
        * bytes_per_param
        * activation_layers
        * 2
        / 1e9
    )

    total = model_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "model_gb": model_memory,
        "optimizer_gb": optimizer_memory,
        "gradient_gb": gradient_memory,
        "activation_gb": activation_memory,
        "total_gb": total,
    }


def auto_find_batch_size(
    model: torch.nn.Module,
    sample_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    start_batch_size: int = 32,
    min_batch_size: int = 1,
) -> int:
    """
    Automatically find the largest batch size that fits in memory.

    Args:
        model: Model to test.
        sample_batch_fn: Function that creates a sample batch given batch size.
        start_batch_size: Initial batch size to try.
        min_batch_size: Minimum acceptable batch size.

    Returns:
        Largest batch size that works.

    Raises:
        RuntimeError: If no batch size >= min_batch_size works.
    """
    batch_size = start_batch_size

    while batch_size >= min_batch_size:
        try:
            clear_memory()

            batch = sample_batch_fn(batch_size)
            outputs = model(**batch)
            outputs.loss.backward()

            model.zero_grad()
            clear_memory()

            logger.info(f"Batch size {batch_size} works!")
            return batch_size

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Batch size {batch_size} OOM, trying smaller...")
            batch_size //= 2
            clear_memory()

    raise RuntimeError(f"Could not find working batch size >= {min_batch_size}")


def get_gpu_utilization() -> Optional[float]:
    """
    Get current GPU utilization percentage.

    Returns:
        GPU utilization as percentage (0-100), or None if unavailable.
    """
    if not torch.cuda.is_available():
        return None

    try:
        # This requires pynvml, fall back gracefully
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return util.gpu
    except (ImportError, Exception):
        return None
