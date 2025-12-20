"""
Custom trainer for Qwen3 encoder-decoder with FSDP2 support.

Uses HuggingFace Accelerate for distributed training, mixed precision,
gradient accumulation, and checkpointing.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

from .config import FullConfig
from .memory_utils import log_memory_stats

# Use standard logger for pre-accelerator messages
_init_logger = logging.getLogger(__name__)
# Accelerate logger for post-accelerator messages
logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Tracks training state for checkpointing."""

    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float = float("inf")
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

    Args:
        model: Model to train.
        tokenizer: Tokenizer instance.
        config: Full training configuration.
        train_dataloader: Training data loader.
        eval_dataloader: Optional evaluation data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: FullConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Enable TF32 for Ampere+ GPUs (up to 3x faster matmuls)
        if getattr(config.training, "use_tf32", True):
            from .optimizations import setup_tf32
            setup_tf32()

        # Apply Liger kernels before accelerator.prepare()
        if getattr(config.training, "use_liger_kernels", True):
            from .optimizations import apply_liger_kernels
            model = apply_liger_kernels(model, model.config)

        # Optional torch.compile (10-20% speedup, slower startup)
        if getattr(config.training, "torch_compile", False):
            _init_logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        # Sync CCE flag from training config to model config
        use_cce = getattr(config.training, "use_cut_cross_entropy", False)
        if hasattr(model, "config"):
            model.config.use_cut_cross_entropy = use_cce

        # Setup accelerator
        project_config = ProjectConfiguration(
            project_dir=config.infra.output_dir,
            logging_dir=config.infra.logging_dir,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=(
                "bf16"
                if config.training.bf16
                else ("fp16" if config.training.fp16 else "no")
            ),
            log_with=config.training.report_to if config.training.report_to else None,
            project_config=project_config,
        )

        # Set seed for reproducibility
        set_seed(config.training.seed)

        # Enable gradient checkpointing if configured
        if config.infra.fsdp_activation_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")

        # Setup optimizer
        self.optimizer = self._create_optimizer(model)

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Prepare with accelerator (handles FSDP wrapping, etc.)
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
            # Log optimization settings
            if use_cce:
                logger.info("Cut Cross Entropy enabled for memory-efficient loss")

    def _create_optimizer(self, model: nn.Module) -> AdamW:
        """Create optimizer with weight decay separation."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and LayerNorm/RMSNorm
            if (
                "bias" in name
                or "layernorm" in name.lower()
                or "norm" in name.lower()
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.training.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW for better CUDA performance
        use_fused = (
            getattr(self.config.training, "use_fused_adamw", True)
            and torch.cuda.is_available()
        )

        return AdamW(
            optimizer_groups,
            lr=self.config.training.learning_rate,
            betas=(
                self.config.training.adam_beta1,
                self.config.training.adam_beta2,
            ),
            eps=self.config.training.adam_epsilon,
            fused=use_fused,
        )

    def _create_scheduler(self) -> SequentialLR:
        """Create learning rate scheduler with warmup."""
        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.training.warmup_steps,
        )

        # Cosine decay
        cosine_steps = (
            self.config.training.num_train_steps - self.config.training.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(cosine_steps, 1),
            eta_min=self.config.training.learning_rate * 0.1,  # 10% of peak LR
        )

        # Combine: warmup then cosine
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.training.warmup_steps],
        )

    def _init_logging(self) -> None:
        """Initialize logging (W&B, TensorBoard)."""
        if self.config.training.report_to:
            try:
                init_kwargs = {
                    "wandb": {
                        "entity": self.config.infra.wandb_entity,
                        "name": self.config.infra.wandb_run_name,
                    }
                }
                # Add tags if specified
                if getattr(self.config.infra, "wandb_tags", None):
                    init_kwargs["wandb"]["tags"] = self.config.infra.wandb_tags

                self.accelerator.init_trackers(
                    project_name=self.config.infra.wandb_project,
                    config=self.config.to_dict(),
                    init_kwargs=init_kwargs,
                )

                # Enable wandb.watch for gradient tracking
                wandb_watch = getattr(self.config.infra, "wandb_watch", None)
                if wandb_watch and "wandb" in self.config.training.report_to:
                    try:
                        import wandb
                        wandb.watch(
                            self.model,
                            log=wandb_watch,
                            log_freq=100,
                        )
                        logger.info(f"Enabled wandb.watch with log={wandb_watch}")
                    except Exception as e:
                        logger.warning(f"Failed to enable wandb.watch: {e}")

            except Exception as e:
                logger.warning(f"Failed to initialize trackers: {e}")

    def train(self) -> None:
        """Run training loop."""
        logger.info("***** Starting training *****")
        logger.info(f"  Num training steps = {self.config.training.num_train_steps}")
        logger.info(
            f"  Instantaneous batch size per device = "
            f"{self.config.training.per_device_train_batch_size}"
        )
        logger.info(
            f"  Gradient accumulation steps = "
            f"{self.config.training.gradient_accumulation_steps}"
        )

        effective_batch_size = (
            self.config.training.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.config.training.gradient_accumulation_steps
        )
        logger.info(f"  Total train batch size = {effective_batch_size}")

        log_memory_stats("Initial: ")

        progress_bar = tqdm(
            range(self.config.training.num_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )

        self.model.train()

        train_loss = 0.0
        step_tokens = 0  # Track tokens for throughput calculation
        step_samples = 0  # Track samples for throughput calculation
        import time
        last_log_time = time.time()

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
                batch_tokens = batch["input_ids"].numel()
                if "decoder_input_ids" in batch:
                    batch_tokens += batch["decoder_input_ids"].numel()
                self.state.total_tokens_seen += (
                    batch_tokens * self.accelerator.num_processes
                )
                step_tokens += batch_tokens * self.accelerator.num_processes
                step_samples += batch["input_ids"].shape[0] * self.accelerator.num_processes

                progress_bar.update(1)

                # Logging
                if self.state.global_step % self.config.training.logging_steps == 0:
                    avg_loss = train_loss / self.config.training.logging_steps

                    # Calculate throughput
                    current_time = time.time()
                    elapsed = current_time - last_log_time
                    tokens_per_sec = step_tokens / elapsed if elapsed > 0 else 0
                    samples_per_sec = step_samples / elapsed if elapsed > 0 else 0

                    # GPU memory stats
                    gpu_memory_gb = 0.0
                    if torch.cuda.is_available():
                        gpu_memory_gb = torch.cuda.max_memory_allocated() / 1e9

                    self.accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": self.state.epoch,
                            "train/tokens_seen": self.state.total_tokens_seen,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/samples_per_sec": samples_per_sec,
                            "train/gpu_memory_gb": gpu_memory_gb,
                        },
                        step=self.state.global_step,
                    )

                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                        tok_s=f"{tokens_per_sec:.0f}",
                    )

                    # Reset counters
                    train_loss = 0.0
                    step_tokens = 0
                    step_samples = 0
                    last_log_time = current_time

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
        """
        Run evaluation.

        Returns:
            Average evaluation loss.
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        max_batches = (
            self.config.training.eval_samples
            // self.config.training.per_device_eval_batch_size
        )

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

            if num_batches >= max_batches:
                break

        avg_loss = total_loss / max(num_batches, 1)

        # Gather across processes
        avg_loss_tensor = torch.tensor([avg_loss]).to(self.accelerator.device)
        avg_loss = self.accelerator.gather(avg_loss_tensor).mean().item()

        return avg_loss

    def _save_checkpoint(self, name: str) -> None:
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
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(output_dir)
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "save_pretrained"
            ):
                self.model.config.save_pretrained(output_dir)

            # Track checkpoint path for averaging
            if name.startswith("step-"):
                self.checkpoint_paths.append(output_dir)

                # Keep only last N checkpoints
                while len(self.checkpoint_paths) > self.config.training.save_total_limit:
                    old_path = self.checkpoint_paths.pop(0)
                    if old_path.exists():
                        shutil.rmtree(old_path)

        logger.info(f"Saved checkpoint to {output_dir}")

    def _average_checkpoints(self, num_checkpoints: int = 5) -> None:
        """Average the last N checkpoints."""
        logger.info(f"Averaging last {num_checkpoints} checkpoints...")

        checkpoints_to_average = self.checkpoint_paths[-num_checkpoints:]

        if len(checkpoints_to_average) < 2:
            logger.warning("Not enough checkpoints to average")
            return

        # Load and average state dicts
        avg_state: Dict[str, torch.Tensor] = {}

        for ckpt_path in checkpoints_to_average:
            model_path = ckpt_path / "pytorch_model.bin"
            if not model_path.exists():
                # Try accelerator's default path
                model_path = ckpt_path / "model.safetensors"
            if not model_path.exists():
                logger.warning(f"No model file found in {ckpt_path}")
                continue

            state_dict = torch.load(model_path, map_location="cpu")

            for key, value in state_dict.items():
                if key not in avg_state:
                    avg_state[key] = value.float() / len(checkpoints_to_average)
                else:
                    avg_state[key] += value.float() / len(checkpoints_to_average)

        if not avg_state:
            logger.warning("No state dicts loaded for averaging")
            return

        # Convert back to original dtype
        for key in avg_state:
            avg_state[key] = avg_state[key].to(torch.bfloat16)

        # Save averaged checkpoint
        output_dir = Path(self.config.infra.output_dir) / "checkpoint-averaged"
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(avg_state, output_dir / "pytorch_model.bin")

        # Copy config and tokenizer from last checkpoint
        last_ckpt = checkpoints_to_average[-1]
        for pattern in ["config.json", "tokenizer*", "sentinel_config.json"]:
            for f in last_ckpt.glob(pattern):
                shutil.copy(f, output_dir / f.name)

        logger.info(f"Saved averaged checkpoint to {output_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        self.accelerator.load_state(checkpoint_path)

        # Load training state
        state_path = Path(checkpoint_path) / "training_state.pt"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu")
            self.state.global_step = state_dict["global_step"]
            self.state.epoch = state_dict["epoch"]
            self.state.best_eval_loss = state_dict["best_eval_loss"]
            self.state.total_tokens_seen = state_dict["total_tokens_seen"]

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} (step {self.state.global_step})"
        )
