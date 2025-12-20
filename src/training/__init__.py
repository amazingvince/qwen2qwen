"""Training infrastructure for Qwen3 Encoder-Decoder."""

from .config import (
    DataConfig,
    FullConfig,
    InfraConfig,
    ModelConfig,
    TrainingConfig,
)
from .execution import (
    PHASE_CONFIGS,
    PhaseConfig,
    TrainingPhase,
    estimate_steps_for_tokens,
    estimate_training_time,
)
from .memory_utils import (
    auto_find_batch_size,
    clear_memory,
    estimate_model_memory,
    get_memory_stats,
    log_memory_stats,
)
from .monitor import (
    MetricWindow,
    TrainingMonitor,
    compute_gradient_norm,
)
from .trainer import Qwen3EncoderDecoderTrainer, TrainingState

__all__ = [
    # Configuration
    "FullConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "InfraConfig",
    # Training phases
    "TrainingPhase",
    "PhaseConfig",
    "PHASE_CONFIGS",
    "estimate_steps_for_tokens",
    "estimate_training_time",
    # Monitoring
    "MetricWindow",
    "TrainingMonitor",
    "compute_gradient_norm",
    # Trainer
    "Qwen3EncoderDecoderTrainer",
    "TrainingState",
    # Memory utilities
    "get_memory_stats",
    "log_memory_stats",
    "clear_memory",
    "estimate_model_memory",
    "auto_find_batch_size",
]
