"""
Training execution phases and validation utilities.

Defines training phases for progressive scaling from validation to full training.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phase definitions."""

    SANITY_CHECK = "sanity_check"  # Few steps to verify forward/backward
    VALIDATION = "validation"  # 1B tokens
    MEDIUM = "medium"  # 50-100B tokens
    FULL = "full"  # 500B+ tokens


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""

    phase: TrainingPhase
    num_tokens: int
    description: str

    # Override training config
    num_train_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    logging_steps: Optional[int] = None

    # Validation criteria
    max_loss_after_warmup: float = 10.0  # Fail if loss exceeds this
    min_loss_decrease: float = 0.1  # Minimum loss decrease over phase


# Default phase configurations
PHASE_CONFIGS: Dict[TrainingPhase, PhaseConfig] = {
    TrainingPhase.SANITY_CHECK: PhaseConfig(
        phase=TrainingPhase.SANITY_CHECK,
        num_tokens=10_000_000,  # 10M tokens
        description="Quick sanity check - verify forward/backward pass",
        num_train_steps=100,
        eval_steps=50,
        save_steps=100,
        logging_steps=1,
        max_loss_after_warmup=15.0,
    ),
    TrainingPhase.VALIDATION: PhaseConfig(
        phase=TrainingPhase.VALIDATION,
        num_tokens=1_000_000_000,  # 1B tokens
        description="Validation run - verify training converges",
        num_train_steps=5000,
        eval_steps=500,
        save_steps=1000,
        logging_steps=10,
        max_loss_after_warmup=8.0,
        min_loss_decrease=0.5,
    ),
    TrainingPhase.MEDIUM: PhaseConfig(
        phase=TrainingPhase.MEDIUM,
        num_tokens=100_000_000_000,  # 100B tokens
        description="Medium training run",
        num_train_steps=50000,
        eval_steps=1000,
        save_steps=5000,
        logging_steps=10,
        max_loss_after_warmup=5.0,
        min_loss_decrease=1.0,
    ),
    TrainingPhase.FULL: PhaseConfig(
        phase=TrainingPhase.FULL,
        num_tokens=500_000_000_000,  # 500B tokens
        description="Full production training",
        num_train_steps=100000,
        eval_steps=500,
        save_steps=10000,
        logging_steps=10,
        max_loss_after_warmup=4.0,
        min_loss_decrease=1.5,
    ),
}


def estimate_steps_for_tokens(
    num_tokens: int,
    batch_size: int,
    seq_length: int,
    gradient_accumulation_steps: int,
    num_gpus: int,
) -> int:
    """
    Estimate training steps for a given number of tokens.

    Args:
        num_tokens: Target number of tokens
        batch_size: Per-device batch size
        seq_length: Average sequence length (encoder + decoder)
        gradient_accumulation_steps: Gradient accumulation
        num_gpus: Number of GPUs

    Returns:
        Estimated number of training steps
    """
    tokens_per_step = batch_size * seq_length * gradient_accumulation_steps * num_gpus
    return num_tokens // tokens_per_step


def estimate_training_time(
    num_steps: int,
    step_time_seconds: float,
) -> Dict[str, float]:
    """
    Estimate total training time.

    Args:
        num_steps: Number of training steps
        step_time_seconds: Seconds per step

    Returns:
        Dictionary with time estimates
    """
    total_seconds = num_steps * step_time_seconds

    return {
        "seconds": total_seconds,
        "minutes": total_seconds / 60,
        "hours": total_seconds / 3600,
        "days": total_seconds / 86400,
    }
