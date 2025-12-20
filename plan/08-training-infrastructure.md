# Story 08: Training Infrastructure with Accelerate & FSDP2

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-08 |
| **Title** | Training Infrastructure with HuggingFace Accelerate and FSDP2 |
| **Epic** | Qwen3 Encoder-Decoder Implementation |
| **Priority** | High |
| **Estimated Effort** | 3-4 days |
| **Dependencies** | Stories 01-07 (Full model and data pipeline) |

---

## Objective

Set up distributed training infrastructure using HuggingFace Accelerate with FSDP2 (Fully Sharded Data Parallel v2) for efficient multi-GPU training of the ~1B parameter encoder-decoder model. This includes configuration, memory optimization, logging, and checkpointing.

---

## Background

### Why FSDP2?

FSDP2 provides significant improvements over FSDP1:
- **Per-parameter sharding** with DTensor for better memory efficiency
- **Support for mixed precision** including FP8
- **Async checkpointing** for faster saves
- **Better composability** with other parallelism strategies

### Memory Requirements

For a ~1B parameter model with AdamW:
- Model parameters: ~2GB (BF16)
- Optimizer states: ~8GB (FP32 moments)
- Gradients: ~2GB (BF16)
- Activations: Variable (depends on batch size and sequence length)

FSDP2 shards all of these across GPUs, enabling training on hardware that couldn't fit the full model.

---

## Technical Requirements

### 8.1 Accelerate Configuration

```yaml
# configs/accelerate_fsdp2.yaml

compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'

fsdp_config:
  # Use FSDP2
  fsdp_version: 2
  
  # Auto-wrap transformer layers
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap:
    - Qwen3EncoderLayer
    - Qwen3DecoderLayer
  
  # Sharding strategy
  fsdp_sharding_strategy: FULL_SHARD  # Shard everything
  
  # State dict handling
  fsdp_state_dict_type: SHARDED_STATE_DICT
  
  # Memory optimization
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false  # Set true if OOM
  
  # Activation checkpointing
  fsdp_activation_checkpointing: true
  
  # Sync module states at init
  fsdp_sync_module_states: true
  
  # Use original params for optimizer
  fsdp_use_orig_params: true
  
  # Backward prefetch for overlapping communication
  fsdp_backward_prefetch: BACKWARD_PRE

# Mixed precision
mixed_precision: bf16

# Machine config
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 8  # Adjust based on GPU count

# Misc
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

```yaml
# configs/accelerate_deepspeed.yaml
# Alternative: DeepSpeed ZeRO-3 configuration

compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED

deepspeed_config:
  deepspeed_multinode_launcher: standard
  
  # ZeRO optimization
  zero_stage: 3
  
  # Offloading for memory efficiency
  offload_optimizer_device: cpu
  offload_param_device: none  # Set to 'cpu' if needed
  
  # Gradient handling
  gradient_accumulation_steps: auto
  gradient_clipping: 1.0
  
  # ZeRO-3 specific
  zero3_init_flag: true
  zero3_save_16bit_model: true

mixed_precision: bf16
num_machines: 1
num_processes: 8
```

### 8.2 Training Configuration

```python
# src/training/config.py

"""
Training configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


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
    
    # UL2 specific
    ul2_task_weights: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 4])
    
    # Preprocessing
    preprocessing_num_workers: int = 4
    shuffle_buffer_size: int = 10000


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


@dataclass
class FullConfig:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "FullConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            infra=InfraConfig(**config_dict.get("infra", {})),
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
```

### 8.3 Trainer Class

```python
# src/training/trainer.py

"""
Custom trainer for Qwen3 encoder-decoder with FSDP2 support.
"""

import os
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import wandb

from ..modeling_qwen3_encdec import Qwen3ForSeq2SeqLM
from ..tokenization_qwen3_encdec import Qwen3EncoderDecoderTokenizer
from ..data.collator import UL2DataCollator
from .config import FullConfig

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Tracks training state for checkpointing."""
    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float = float('inf')
    total_tokens_seen: int = 0


class Qwen3EncoderDecoderTrainer:
    """
    Trainer for Qwen3 encoder-decoder model.
    
    Features:
    - FSDP2 distributed training via Accelerate
    - Mixed precision (BF16)
    - Gradient checkpointing
    - Checkpoint averaging
    - W&B logging
    """
    
    def __init__(
        self,
        model: Qwen3ForSeq2SeqLM,
        tokenizer: Qwen3EncoderDecoderTokenizer,
        config: FullConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup accelerator
        project_config = ProjectConfiguration(
            project_dir=config.infra.output_dir,
            logging_dir=config.infra.logging_dir,
        )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision="bf16" if config.training.bf16 else ("fp16" if config.training.fp16 else "no"),
            log_with=config.training.report_to,
            project_config=project_config,
        )
        
        # Set seed for reproducibility
        set_seed(config.training.seed)
        
        # Enable gradient checkpointing if using FSDP
        if config.infra.fsdp_activation_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer(model)
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model,
            self.optimizer,
            train_dataloader,
            self.scheduler,
        )
        
        if eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(eval_dataloader)
        else:
            self.eval_dataloader = None
        
        # Training state
        self.state = TrainingState()
        
        # Checkpoint paths for averaging
        self.checkpoint_paths: List[Path] = []
        
        # Initialize logging
        if self.accelerator.is_main_process:
            self._init_logging()
    
    def _create_optimizer(self, model: nn.Module) -> AdamW:
        """Create optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and LayerNorm
            if "bias" in name or "layernorm" in name.lower() or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            optimizer_groups,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon,
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=self.config.training.warmup_steps,
        )
        
        # Cosine decay
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_train_steps - self.config.training.warmup_steps,
            eta_min=self.config.training.learning_rate * 0.1,  # 10% of peak LR
        )
        
        # Combine: warmup then cosine
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.training.warmup_steps],
        )
    
    def _init_logging(self):
        """Initialize logging (W&B, TensorBoard)."""
        if "wandb" in self.config.training.report_to:
            self.accelerator.init_trackers(
                project_name=self.config.infra.wandb_project,
                config=self.config.__dict__,
                init_kwargs={
                    "wandb": {
                        "entity": self.config.infra.wandb_entity,
                        "name": self.config.infra.wandb_run_name,
                    }
                },
            )
    
    def train(self):
        """Run training loop."""
        logger.info("***** Starting training *****")
        logger.info(f"  Num training steps = {self.config.training.num_train_steps}")
        logger.info(f"  Instantaneous batch size per device = {self.config.training.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.training.gradient_accumulation_steps}")
        
        effective_batch_size = (
            self.config.training.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.config.training.gradient_accumulation_steps
        )
        logger.info(f"  Total train batch size = {effective_batch_size}")
        
        progress_bar = tqdm(
            range(self.config.training.num_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )
        
        self.model.train()
        
        train_loss = 0.0
        
        # Infinite iterator for streaming datasets
        train_iterator = iter(self.train_dataloader)
        
        while self.state.global_step < self.config.training.num_train_steps:
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
                self.state.epoch += 1
            
            # Forward pass
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update tracking
            if self.accelerator.sync_gradients:
                train_loss += loss.detach().item()
                self.state.global_step += 1
                
                # Update tokens seen
                batch_tokens = batch["input_ids"].numel() + batch["decoder_input_ids"].numel()
                self.state.total_tokens_seen += batch_tokens * self.accelerator.num_processes
                
                progress_bar.update(1)
                
                # Logging
                if self.state.global_step % self.config.training.logging_steps == 0:
                    avg_loss = train_loss / self.config.training.logging_steps
                    
                    self.accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": self.state.epoch,
                            "train/tokens_seen": self.state.total_tokens_seen,
                        },
                        step=self.state.global_step,
                    )
                    
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    )
                    
                    train_loss = 0.0
                
                # Evaluation
                if (
                    self.eval_dataloader is not None
                    and self.state.global_step % self.config.training.eval_steps == 0
                ):
                    eval_loss = self.evaluate()
                    
                    self.accelerator.log(
                        {"eval/loss": eval_loss},
                        step=self.state.global_step,
                    )
                    
                    if eval_loss < self.state.best_eval_loss:
                        self.state.best_eval_loss = eval_loss
                        self._save_checkpoint("best")
                    
                    self.model.train()
                
                # Checkpointing
                if self.state.global_step % self.config.training.save_steps == 0:
                    self._save_checkpoint(f"step-{self.state.global_step}")
        
        # Final save
        self._save_checkpoint("final")
        
        # Checkpoint averaging
        if self.accelerator.is_main_process and len(self.checkpoint_paths) >= 3:
            self._average_checkpoints()
        
        self.accelerator.end_training()
        logger.info("Training complete!")
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
            
            if num_batches >= self.config.training.eval_samples // self.config.training.per_device_eval_batch_size:
                break
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Gather across processes
        avg_loss = self.accelerator.gather(torch.tensor([avg_loss]).to(self.accelerator.device)).mean().item()
        
        return avg_loss
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        self.accelerator.wait_for_everyone()
        
        output_dir = Path(self.config.infra.output_dir) / f"checkpoint-{name}"
        
        # Save with accelerator (handles FSDP state dict)
        self.accelerator.save_state(str(output_dir))
        
        # Save additional state
        if self.accelerator.is_main_process:
            # Save training state
            torch.save(
                {
                    "global_step": self.state.global_step,
                    "epoch": self.state.epoch,
                    "best_eval_loss": self.state.best_eval_loss,
                    "total_tokens_seen": self.state.total_tokens_seen,
                },
                output_dir / "training_state.pt",
            )
            
            # Save tokenizer and config
            self.tokenizer.save_pretrained(output_dir)
            self.model.config.save_pretrained(output_dir)
            
            # Track checkpoint path for averaging
            if name.startswith("step-"):
                self.checkpoint_paths.append(output_dir)
                
                # Keep only last N checkpoints
                if len(self.checkpoint_paths) > self.config.training.save_total_limit:
                    old_path = self.checkpoint_paths.pop(0)
                    # Optionally delete old checkpoint
        
        logger.info(f"Saved checkpoint to {output_dir}")
    
    def _average_checkpoints(self, num_checkpoints: int = 5):
        """Average the last N checkpoints."""
        logger.info(f"Averaging last {num_checkpoints} checkpoints...")
        
        checkpoints_to_average = self.checkpoint_paths[-num_checkpoints:]
        
        if len(checkpoints_to_average) < 2:
            logger.warning("Not enough checkpoints to average")
            return
        
        # Load and average state dicts
        avg_state = {}
        
        for i, ckpt_path in enumerate(checkpoints_to_average):
            # Load checkpoint
            state_dict = torch.load(ckpt_path / "pytorch_model.bin", map_location="cpu")
            
            for key, value in state_dict.items():
                if key not in avg_state:
                    avg_state[key] = value.float() / len(checkpoints_to_average)
                else:
                    avg_state[key] += value.float() / len(checkpoints_to_average)
        
        # Convert back to original dtype
        for key in avg_state:
            avg_state[key] = avg_state[key].to(torch.bfloat16)
        
        # Save averaged checkpoint
        output_dir = Path(self.config.infra.output_dir) / "checkpoint-averaged"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(avg_state, output_dir / "pytorch_model.bin")
        
        # Copy config and tokenizer from last checkpoint
        import shutil
        last_ckpt = checkpoints_to_average[-1]
        shutil.copy(last_ckpt / "config.json", output_dir / "config.json")
        for f in last_ckpt.glob("tokenizer*"):
            shutil.copy(f, output_dir / f.name)
        
        logger.info(f"Saved averaged checkpoint to {output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        
        # Load training state
        state_path = Path(checkpoint_path) / "training_state.pt"
        if state_path.exists():
            state_dict = torch.load(state_path)
            self.state.global_step = state_dict["global_step"]
            self.state.epoch = state_dict["epoch"]
            self.state.best_eval_loss = state_dict["best_eval_loss"]
            self.state.total_tokens_seen = state_dict["total_tokens_seen"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (step {self.state.global_step})")
```

### 8.4 Memory Optimization Utilities

```python
# src/training/memory_utils.py

"""
Memory optimization utilities for training.
"""

import gc
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def log_memory_stats(prefix: str = ""):
    """Log current memory statistics."""
    stats = get_memory_stats()
    if stats.get("gpu_available", True):
        logger.info(
            f"{prefix}Memory: allocated={stats['allocated_gb']:.2f}GB, "
            f"reserved={stats['reserved_gb']:.2f}GB, "
            f"max={stats['max_allocated_gb']:.2f}GB"
        )


def clear_memory():
    """Clear GPU memory cache."""
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
) -> dict:
    """
    Estimate memory requirements for training.
    
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    
    # Model parameters
    model_memory = num_params * bytes_per_param / 1e9
    
    # Optimizer states (AdamW: 2 FP32 moments per param)
    if optimizer == "adamw":
        optimizer_memory = num_params * 4 * 2 / 1e9  # 2 FP32 states
    else:
        optimizer_memory = 0
    
    # Gradients
    gradient_memory = num_params * bytes_per_param / 1e9
    
    # Activations (rough estimate)
    # Each layer stores activations of size: batch * seq * hidden * 2 (forward + backward)
    if gradient_checkpointing:
        # With checkpointing, only sqrt(layers) activations stored
        activation_layers = int(num_layers ** 0.5)
    else:
        activation_layers = num_layers
    
    activation_memory = (
        batch_size * seq_length * hidden_size * bytes_per_param * activation_layers * 2 / 1e9
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
    model,
    sample_batch_fn,
    start_batch_size: int = 32,
    min_batch_size: int = 1,
) -> int:
    """
    Automatically find the largest batch size that fits in memory.
    
    Args:
        model: Model to test
        sample_batch_fn: Function that creates a sample batch given batch size
        start_batch_size: Initial batch size to try
        min_batch_size: Minimum acceptable batch size
        
    Returns:
        Largest batch size that works
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
```

### 8.5 Launch Script

```bash
#!/bin/bash
# scripts/launch_training.sh

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="qwen3-encoder-decoder"

# Accelerate configuration
ACCELERATE_CONFIG="configs/accelerate_fsdp2.yaml"

# Training configuration
TRAIN_CONFIG="configs/training_config.yaml"

# Output directory
OUTPUT_DIR="./output/qwen3-encdec-$(date +%Y%m%d_%H%M%S)"

# Launch training
accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    scripts/train.py \
    --config $TRAIN_CONFIG \
    --output_dir $OUTPUT_DIR \
    "$@"
```

```python
# scripts/train.py

"""
Main training script.

Usage:
    accelerate launch scripts/train.py --config configs/training.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.config import FullConfig
from src.training.trainer import Qwen3EncoderDecoderTrainer
from src.weight_initialization import initialize_from_qwen3
from src.tokenization_qwen3_encdec import Qwen3EncoderDecoderTokenizer
from src.data.ul2_dataset import create_ul2_dataset
from src.data.collator import UL2DataCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = FullConfig.from_yaml(args.config)
    
    if args.output_dir:
        config.infra.output_dir = args.output_dir
    
    # Create output directory
    Path(config.infra.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config copy
    config.to_yaml(Path(config.infra.output_dir) / "config.yaml")
    
    logger.info("Initializing model from Qwen3 checkpoint...")
    model = initialize_from_qwen3(
        config.model.model_name_or_path,
        num_sentinel_tokens=config.model.num_sentinel_tokens,
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        num_sentinel_tokens=config.model.num_sentinel_tokens,
    )
    
    logger.info("Creating datasets...")
    train_dataset = create_ul2_dataset(
        config.data.dataset_name,
        tokenizer,
        streaming=config.data.streaming,
        max_seq_length=config.data.max_seq_length,
    )
    
    # Create data collator
    collator = UL2DataCollator(
        tokenizer,
        max_encoder_length=config.data.max_encoder_length,
        max_decoder_length=config.data.max_decoder_length,
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        collate_fn=collator,
        num_workers=config.data.preprocessing_num_workers,
        pin_memory=True,
    )
    
    # Create trainer
    trainer = Qwen3EncoderDecoderTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataloader=train_dataloader,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
```

---

## Unit Tests

```python
# tests/test_training_infra.py

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.training.config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    InfraConfig,
    FullConfig,
)
from src.training.memory_utils import (
    get_memory_stats,
    estimate_model_memory,
    clear_memory,
)


class TestConfigurations:
    """Tests for configuration classes."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        assert config.model_name_or_path == "Qwen/Qwen3-0.6B"
        assert config.num_sentinel_tokens == 100
        assert config.tokenizer_name_or_path == config.model_name_or_path
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        assert config.learning_rate == 2e-4
        assert config.max_grad_norm == 1.0
        assert config.bf16 is True
    
    def test_full_config_to_yaml(self):
        """Test saving config to YAML."""
        config = FullConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(path))
            
            assert path.exists()
            
            # Load and verify
            loaded = FullConfig.from_yaml(str(path))
            assert loaded.model.model_name_or_path == config.model.model_name_or_path
            assert loaded.training.learning_rate == config.training.learning_rate
    
    def test_effective_batch_size_calculation(self):
        """Test effective batch size calculation."""
        config = TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
        )
        
        num_gpus = 8
        effective_batch = (
            config.per_device_train_batch_size
            * num_gpus
            * config.gradient_accumulation_steps
        )
        
        assert effective_batch == 256


class TestMemoryUtils:
    """Tests for memory utilities."""
    
    def test_estimate_model_memory(self):
        """Test memory estimation."""
        estimates = estimate_model_memory(
            num_params=1_000_000_000,  # 1B params
            batch_size=4,
            seq_length=8192,
            hidden_size=1024,
            num_layers=28,
            dtype=torch.bfloat16,
            optimizer="adamw",
            gradient_checkpointing=True,
        )
        
        assert "model_gb" in estimates
        assert "optimizer_gb" in estimates
        assert "total_gb" in estimates
        
        # Sanity checks
        assert estimates["model_gb"] > 0
        assert estimates["optimizer_gb"] > estimates["model_gb"]  # Optimizer states larger
        assert estimates["total_gb"] > estimates["model_gb"]
    
    def test_estimate_model_memory_no_checkpointing(self):
        """Test memory estimation without gradient checkpointing."""
        with_ckpt = estimate_model_memory(
            num_params=1_000_000_000,
            batch_size=4,
            seq_length=2048,
            hidden_size=1024,
            num_layers=28,
            gradient_checkpointing=True,
        )
        
        without_ckpt = estimate_model_memory(
            num_params=1_000_000_000,
            batch_size=4,
            seq_length=2048,
            hidden_size=1024,
            num_layers=28,
            gradient_checkpointing=False,
        )
        
        # Without checkpointing should use more memory
        assert without_ckpt["activation_gb"] > with_ckpt["activation_gb"]


class TestTrainerComponents:
    """Tests for trainer components."""
    
    def test_optimizer_param_groups(self):
        """Test optimizer creates correct parameter groups."""
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.LayerNorm(10),
            torch.nn.Linear(10, 10),
        )
        
        # Separate params like trainer does
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # LayerNorm weight and all biases should be in no_decay
        assert len(no_decay_params) == 3  # 2 biases + 1 LN weight
        assert len(decay_params) == 3  # 2 Linear weights + 1 LN (actually just 2)
    
    def test_scheduler_warmup(self):
        """Test learning rate scheduler warmup."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        from torch.optim.lr_scheduler import LinearLR
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=100,
        )
        
        # Initial LR should be low
        assert warmup_scheduler.get_last_lr()[0] < 1e-4
        
        # After warmup, should be at full LR
        for _ in range(100):
            warmup_scheduler.step()
        
        assert abs(warmup_scheduler.get_last_lr()[0] - 1e-4) < 1e-8


class TestCheckpointing:
    """Tests for checkpoint functionality."""
    
    def test_checkpoint_save_load_state(self):
        """Test saving and loading training state."""
        from src.training.trainer import TrainingState
        
        state = TrainingState(
            global_step=1000,
            epoch=2,
            best_eval_loss=0.5,
            total_tokens_seen=1_000_000,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.pt"
            
            torch.save({
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_eval_loss": state.best_eval_loss,
                "total_tokens_seen": state.total_tokens_seen,
            }, path)
            
            loaded = torch.load(path)
            
            assert loaded["global_step"] == 1000
            assert loaded["epoch"] == 2
            assert loaded["best_eval_loss"] == 0.5
    
    def test_checkpoint_averaging_math(self):
        """Test checkpoint averaging produces correct values."""
        # Simulate 3 checkpoints
        ckpts = [
            {"weight": torch.tensor([1.0, 2.0, 3.0])},
            {"weight": torch.tensor([2.0, 3.0, 4.0])},
            {"weight": torch.tensor([3.0, 4.0, 5.0])},
        ]
        
        # Average
        avg_state = {}
        for ckpt in ckpts:
            for key, value in ckpt.items():
                if key not in avg_state:
                    avg_state[key] = value / len(ckpts)
                else:
                    avg_state[key] += value / len(ckpts)
        
        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(avg_state["weight"], expected)
```

---

## Acceptance Criteria

1. **Configuration**
   - [ ] FSDP2 configuration works with Accelerate
   - [ ] All hyperparameters configurable via YAML
   - [ ] Configuration can be saved and loaded

2. **Distributed Training**
   - [ ] Model shards correctly across GPUs with FSDP2
   - [ ] Gradient accumulation works correctly
   - [ ] Mixed precision (BF16) training works

3. **Memory Optimization**
   - [ ] Gradient checkpointing reduces memory usage
   - [ ] Training fits on 8x A100 80GB GPUs
   - [ ] No OOM errors during training

4. **Checkpointing**
   - [ ] Checkpoints save correctly with FSDP2
   - [ ] Training can resume from checkpoint
   - [ ] Checkpoint averaging works

5. **Logging**
   - [ ] W&B logging works
   - [ ] Loss, LR, and other metrics logged
   - [ ] Memory statistics tracked

6. **Launch**
   - [ ] Launch script works correctly
   - [ ] Multi-node training supported
   - [ ] Environment variables properly set

---

## Dependencies

- **Stories 01-05**: Model implementation
- **Story 06**: Weight initialization
- **Story 07**: UL2 data pipeline

---

## Estimated Effort

- Configuration and setup: 1 day
- Trainer implementation: 1.5 days
- Memory optimization: 0.5 days
- Testing and debugging: 1 day
- **Total: 3-4 days**

---

## Developer Notes

1. **FSDP2 vs FSDP1**: FSDP2 requires `fsdp_version: 2` in config. It uses DTensor internally for better memory efficiency.

2. **Gradient Checkpointing**: Essential for long sequences. Trades compute for memory by recomputing activations during backward.

3. **Mixed Precision**: BF16 is preferred over FP16 for stability. It has the same exponent range as FP32.

4. **Checkpoint Averaging**: Averaging the last 5 checkpoints reduces variance and often improves final quality.

5. **DeepSpeed Alternative**: If FSDP2 has issues, DeepSpeed ZeRO-3 is a solid fallback with similar functionality.

6. **Debugging Tips**:
   - Start with 1 GPU to debug before scaling
   - Use `CUDA_LAUNCH_BLOCKING=1` for better error messages
   - Monitor memory with `nvidia-smi -l 1`

7. **Reference Implementations**:
   - HuggingFace Accelerate examples
   - PyTorch FSDP2 documentation
   - T5Gemma 2 training code (if available)
