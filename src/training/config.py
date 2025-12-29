"""Training configuration dataclasses."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    tokenizer_name_or_path: Optional[str] = None
    num_sentinel_tokens: int = 100
    use_flash_attention: bool = True

    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: Optional[str] = None
    text_column: str = "text"
    streaming: bool = True

    max_seq_length: int = 8192
    max_encoder_length: int = 4096
    max_decoder_length: int = 2048

    # UL2 specific - T5Gemma 2 task weights (R1, R2, X1, X2, S)
    ul2_task_weights: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 4])
    # UL2_5 options
    ul2_length_adaptive: bool = False
    ul2_boundary_snapping: bool = False
    ul2_curriculum_start: Optional[List[float]] = None
    ul2_curriculum_end: Optional[List[float]] = None

    # Preprocessing
    preprocessing_num_workers: int = 4
    shuffle_buffer_size: int = 10000
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    dataloader_collate_on_cpu: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Batch size
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8

    # Effective batch size = per_device * num_gpus * gradient_accumulation
    # With 8 GPUs: 4 * 8 * 8 = 256 sequences per step
    # At 8K tokens/seq: ~2M tokens per step

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    num_train_steps: int = 100000
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine"

    # Precision
    bf16: bool = True
    fp16: bool = False

    # GPU Optimizations
    use_tf32: bool = True  # Enable TF32 for matmuls (Ampere+)
    use_liger_kernels: bool = True  # Use Liger optimized kernels
    use_cut_cross_entropy: bool = True  # Use Apple CCE for memory efficiency
    torch_compile: bool = False  # torch.compile (slower startup, 10-20% speedup)
    use_fused_adamw: bool = True  # Use CUDA fused AdamW (faster)

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5

    # Evaluation
    eval_steps: int = 500
    eval_samples: int = 1000

    # Logging
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["wandb", "tensorboard"])

    # Reproducibility
    seed: int = 42


@dataclass
class InfraConfig:
    """Infrastructure configuration."""

    output_dir: str = "./output"
    logging_dir: str = "./logs"

    # Distributed training
    distributed_type: str = "fsdp"  # fsdp, deepspeed, or ddp

    # FSDP specific
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False

    # DeepSpeed specific
    deepspeed_config: Optional[str] = None

    # Hardware
    num_gpus: int = 8

    # Weights & Biases
    wandb_project: str = "qwen3-encoder-decoder"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_watch: Optional[str] = "gradients"  # "gradients", "all", or None
    wandb_log_model: bool = False  # Log checkpoints to W&B
    wandb_tags: Optional[List[str]] = None  # Tags for organization


@dataclass
class FullConfig:
    """Complete training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "FullConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            FullConfig instance.
        """
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            infra=InfraConfig(**config_dict.get("infra", {})),
        )

    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML config.
        """
        config_dict = {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "training": asdict(self.training),
            "infra": asdict(self.infra),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary."""
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "training": asdict(self.training),
            "infra": asdict(self.infra),
        }
