"""
Training optimizations: Liger kernels, TF32, CCE.

This module provides GPU optimization utilities for training:
- TF32 enablement for Ampere+ GPUs
- Liger kernel patching for RMSNorm and SwiGLU MLP
- Cut Cross Entropy (CCE) helpers
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_tf32() -> None:
    """
    Enable TF32 for Ampere+ GPUs.

    TensorFloat-32 uses 10-bit mantissa (vs FP32's 23-bit) for up to 3x
    faster matrix multiplications with minimal precision loss.

    This is a global setting that affects all CUDA operations.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for CUDA matmuls and cuDNN")
    else:
        logger.warning("CUDA not available, TF32 not enabled")


def apply_liger_kernels(model: nn.Module, config) -> nn.Module:
    """
    Apply Liger kernel optimizations to encoder/decoder layers.

    Replaces:
    - Qwen3RMSNorm -> LigerRMSNorm (fused normalization)
    - Qwen3MLP -> LigerSwiGLUMLP (fused gate+activation+multiply)

    Does NOT replace CrossEntropy (using Apple CCE instead).

    Args:
        model: The model to optimize.
        config: Model configuration with hidden_size, intermediate_size, etc.

    Returns:
        The optimized model (modified in-place).
    """
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    except ImportError as exc:
        logger.warning(
            "Liger kernels requested but `liger_kernel` is not installed. "
            "Install the optional 'optimizations' extra to enable. "
            "Skipping Liger kernel replacements.",
            exc_info=exc,
        )
        return model

    counts: Dict[str, int] = {"rms_norm": 0, "mlp": 0}

    for name, module in list(model.named_modules()):
        parent = _get_parent_module(model, name)
        attr = name.split(".")[-1]

        # Replace RMSNorm with LigerRMSNorm
        if module.__class__.__name__ == "Qwen3RMSNorm":
            eps = getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6))
            new_norm = LigerRMSNorm(module.weight.shape[0], eps=eps)
            new_norm.weight = module.weight
            setattr(parent, attr, new_norm)
            counts["rms_norm"] += 1

        # Replace MLP with LigerSwiGLUMLP
        elif module.__class__.__name__ == "Qwen3MLP":
            new_mlp = LigerSwiGLUMLP(config)
            # Copy weights from original MLP
            new_mlp.gate_proj.weight = module.gate_proj.weight
            new_mlp.up_proj.weight = module.up_proj.weight
            new_mlp.down_proj.weight = module.down_proj.weight
            setattr(parent, attr, new_mlp)
            counts["mlp"] += 1

    logger.info(
        f"Liger kernel replacements: "
        f"{counts['rms_norm']} RMSNorm, {counts['mlp']} MLP layers"
    )
    return model


def _get_parent_module(model: nn.Module, name: str) -> nn.Module:
    """Get parent module by dot-separated name path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def check_cce_available() -> bool:
    """Check if Cut Cross Entropy is available."""
    try:
        from cut_cross_entropy import linear_cross_entropy  # noqa: F401

        return True
    except ImportError:
        return False


def get_cce_info() -> Dict[str, str]:
    """Get information about CCE availability and requirements."""
    info = {
        "available": str(check_cce_available()),
        "requirements": "PyTorch 2.4+, Triton 3.0+, Ampere+ GPU",
        "install": "pip install 'cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git'",
    }
    return info
