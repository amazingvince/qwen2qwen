# Story 09: Training Execution Script

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-09 |
| **Title** | Training Execution with Monitoring and Validation |
| **Epic** | Qwen3 Encoder-Decoder Implementation |
| **Priority** | High |
| **Estimated Effort** | 2-3 days |
| **Dependencies** | Stories 01-08 (Full infrastructure) |

---

## Objective

Create a production-ready training execution script with comprehensive monitoring, validation runs, and progressive scaling from small validation to full training. This includes sanity checks, ablation tracking, and training stability monitoring.

---

## Background

Training a ~1B parameter encoder-decoder model requires careful execution:

1. **Validation Run**: Small-scale (1B tokens) to verify everything works
2. **Medium Run**: 50-100B tokens to check convergence
3. **Full Run**: 500B-2T tokens for production model

Each phase should include monitoring for:
- Loss curves (should decrease smoothly)
- Gradient norms (should stay bounded)
- Learning rate schedule
- Memory usage
- Throughput (tokens/second)

---

## Technical Requirements

### 9.1 Training Execution Phases

```python
# src/training/execution.py

"""
Training execution phases and validation utilities.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phase definitions."""
    SANITY_CHECK = "sanity_check"  # Few steps to verify forward/backward
    VALIDATION = "validation"      # 1B tokens
    MEDIUM = "medium"              # 50-100B tokens
    FULL = "full"                  # 500B+ tokens


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
    min_loss_decrease: float = 0.1       # Minimum loss decrease over phase


# Default phase configurations
PHASE_CONFIGS = {
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
```

### 9.2 Training Monitor

```python
# src/training/monitor.py

"""
Training monitoring and validation utilities.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np

import torch

logger = logging.getLogger(__name__)


@dataclass
class MetricWindow:
    """Sliding window for metric tracking."""
    window_size: int = 100
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, value: float):
        self.values.append(value)
    
    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return np.mean(self.values)
    
    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return np.std(self.values)
    
    @property
    def min(self) -> float:
        if not self.values:
            return float('inf')
        return min(self.values)
    
    @property
    def max(self) -> float:
        if not self.values:
            return float('-inf')
        return max(self.values)


@dataclass
class TrainingMonitor:
    """
    Monitor training metrics and detect issues.
    
    Tracks:
    - Loss stability (no sudden spikes)
    - Gradient norms (bounded)
    - Learning rate schedule
    - Throughput
    """
    
    # Alert thresholds
    loss_spike_threshold: float = 2.0  # Alert if loss increases by this factor
    grad_norm_threshold: float = 10.0  # Alert if grad norm exceeds this
    
    # Tracking windows
    loss_window: MetricWindow = field(default_factory=MetricWindow)
    grad_norm_window: MetricWindow = field(default_factory=MetricWindow)
    throughput_window: MetricWindow = field(default_factory=MetricWindow)
    
    # History for plotting
    loss_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    step_history: List[int] = field(default_factory=list)
    
    # Alerts
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Initial loss for comparison
    initial_loss: Optional[float] = None
    warmup_complete_loss: Optional[float] = None
    
    def update(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        tokens_per_second: float,
        is_warmup: bool = False,
    ):
        """Update monitor with new metrics."""
        # Track initial loss
        if self.initial_loss is None:
            self.initial_loss = loss
        
        # Track loss after warmup
        if not is_warmup and self.warmup_complete_loss is None:
            self.warmup_complete_loss = loss
        
        # Update windows
        self.loss_window.add(loss)
        self.grad_norm_window.add(grad_norm)
        self.throughput_window.add(tokens_per_second)
        
        # Update history
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.lr_history.append(learning_rate)
        self.step_history.append(step)
        
        # Check for issues
        self._check_loss_spike(step, loss)
        self._check_grad_norm(step, grad_norm)
    
    def _check_loss_spike(self, step: int, loss: float):
        """Check for sudden loss increases."""
        if len(self.loss_window.values) < 10:
            return
        
        recent_mean = self.loss_window.mean
        
        if loss > recent_mean * self.loss_spike_threshold:
            alert = {
                "type": "loss_spike",
                "step": step,
                "value": loss,
                "threshold": recent_mean * self.loss_spike_threshold,
                "message": f"Loss spike detected: {loss:.4f} > {recent_mean * self.loss_spike_threshold:.4f}",
            }
            self.alerts.append(alert)
            logger.warning(alert["message"])
    
    def _check_grad_norm(self, step: int, grad_norm: float):
        """Check for gradient explosion."""
        if grad_norm > self.grad_norm_threshold:
            alert = {
                "type": "grad_explosion",
                "step": step,
                "value": grad_norm,
                "threshold": self.grad_norm_threshold,
                "message": f"High gradient norm: {grad_norm:.4f} > {self.grad_norm_threshold}",
            }
            self.alerts.append(alert)
            logger.warning(alert["message"])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        return {
            "loss": {
                "current": self.loss_window.mean,
                "initial": self.initial_loss,
                "after_warmup": self.warmup_complete_loss,
                "min": self.loss_window.min,
                "max": self.loss_window.max,
                "decrease": (self.initial_loss - self.loss_window.mean) if self.initial_loss else 0,
            },
            "grad_norm": {
                "mean": self.grad_norm_window.mean,
                "std": self.grad_norm_window.std,
                "max": self.grad_norm_window.max,
            },
            "throughput": {
                "mean_tokens_per_sec": self.throughput_window.mean,
            },
            "alerts": len(self.alerts),
            "steps": len(self.step_history),
        }
    
    def validate_phase(self, phase_config) -> Dict[str, Any]:
        """
        Validate training meets phase criteria.
        
        Returns:
            Validation results with pass/fail status
        """
        results = {
            "passed": True,
            "checks": [],
        }
        
        # Check loss after warmup
        if self.warmup_complete_loss is not None:
            if self.warmup_complete_loss > phase_config.max_loss_after_warmup:
                results["passed"] = False
                results["checks"].append({
                    "name": "loss_after_warmup",
                    "passed": False,
                    "message": f"Loss {self.warmup_complete_loss:.4f} exceeds max {phase_config.max_loss_after_warmup}",
                })
            else:
                results["checks"].append({
                    "name": "loss_after_warmup",
                    "passed": True,
                    "message": f"Loss {self.warmup_complete_loss:.4f} within limit",
                })
        
        # Check loss decrease
        if self.initial_loss and self.loss_window.mean:
            loss_decrease = self.initial_loss - self.loss_window.mean
            if loss_decrease < phase_config.min_loss_decrease:
                results["passed"] = False
                results["checks"].append({
                    "name": "loss_decrease",
                    "passed": False,
                    "message": f"Loss decrease {loss_decrease:.4f} below min {phase_config.min_loss_decrease}",
                })
            else:
                results["checks"].append({
                    "name": "loss_decrease",
                    "passed": True,
                    "message": f"Loss decreased by {loss_decrease:.4f}",
                })
        
        # Check for alerts
        critical_alerts = [a for a in self.alerts if a["type"] == "loss_spike"]
        if len(critical_alerts) > 5:
            results["passed"] = False
            results["checks"].append({
                "name": "loss_spikes",
                "passed": False,
                "message": f"Too many loss spikes: {len(critical_alerts)}",
            })
        
        return results


def compute_gradient_norm(model) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5
```

### 9.3 Sanity Check Script

```python
# scripts/sanity_check.py

"""
Sanity check script - verify model forward/backward works.

Usage:
    python scripts/sanity_check.py --model-path ./initialized-model
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.modeling_qwen3_encdec import Qwen3ForSeq2SeqLM
from src.tokenization_qwen3_encdec import Qwen3EncoderDecoderTokenizer
from src.data.ul2_corruption import UL2Corruptor
from src.data.collator import UL2DataCollator
from src.training.monitor import TrainingMonitor, compute_gradient_norm
from src.training.memory_utils import get_memory_stats, clear_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_batch(tokenizer, batch_size=2, enc_len=128, dec_len=64):
    """Create a dummy batch for testing."""
    vocab_size = tokenizer.vocab_size
    
    encoder_input_ids = torch.randint(0, vocab_size, (batch_size, enc_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, dec_len))
    labels = torch.randint(0, vocab_size, (batch_size, dec_len))
    
    # Set some labels to -100 (ignored)
    labels[:, -10:] = -100
    
    return {
        "input_ids": encoder_input_ids,
        "attention_mask": torch.ones_like(encoder_input_ids),
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": torch.ones_like(decoder_input_ids),
        "labels": labels,
    }


def run_sanity_check(model_path: str, device: str = "cuda"):
    """
    Run comprehensive sanity checks on the model.
    
    Checks:
    1. Model loads correctly
    2. Forward pass works
    3. Loss is computed
    4. Backward pass works
    5. Gradients are non-zero
    6. Tied embeddings work
    7. Generation works
    """
    results = {"passed": True, "checks": []}
    
    # 1. Load model
    logger.info("Loading model...")
    try:
        model = Qwen3ForSeq2SeqLM.from_pretrained(model_path)
        model = model.to(device)
        results["checks"].append({"name": "model_load", "passed": True})
    except Exception as e:
        results["checks"].append({"name": "model_load", "passed": False, "error": str(e)})
        results["passed"] = False
        return results
    
    # Load tokenizer
    try:
        tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(model_path)
        results["checks"].append({"name": "tokenizer_load", "passed": True})
    except Exception as e:
        results["checks"].append({"name": "tokenizer_load", "passed": False, "error": str(e)})
        results["passed"] = False
        return results
    
    # 2. Forward pass
    logger.info("Testing forward pass...")
    batch = create_dummy_batch(tokenizer)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    try:
        model.train()
        outputs = model(**batch)
        results["checks"].append({
            "name": "forward_pass", 
            "passed": True,
            "loss": outputs.loss.item(),
        })
    except Exception as e:
        results["checks"].append({"name": "forward_pass", "passed": False, "error": str(e)})
        results["passed"] = False
        return results
    
    # 3. Loss sanity
    loss_value = outputs.loss.item()
    if not (0 < loss_value < 100):
        results["checks"].append({
            "name": "loss_sanity",
            "passed": False,
            "message": f"Loss {loss_value} outside reasonable range",
        })
        results["passed"] = False
    else:
        results["checks"].append({
            "name": "loss_sanity",
            "passed": True,
            "loss": loss_value,
        })
    
    # 4. Backward pass
    logger.info("Testing backward pass...")
    try:
        outputs.loss.backward()
        results["checks"].append({"name": "backward_pass", "passed": True})
    except Exception as e:
        results["checks"].append({"name": "backward_pass", "passed": False, "error": str(e)})
        results["passed"] = False
        return results
    
    # 5. Gradient checks
    logger.info("Checking gradients...")
    
    # Check encoder has gradients
    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    results["checks"].append({
        "name": "encoder_gradients",
        "passed": encoder_has_grad,
    })
    if not encoder_has_grad:
        results["passed"] = False
    
    # Check decoder has gradients
    decoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.decoder.parameters()
    )
    results["checks"].append({
        "name": "decoder_gradients",
        "passed": decoder_has_grad,
    })
    if not decoder_has_grad:
        results["passed"] = False
    
    # Check gradient norm
    grad_norm = compute_gradient_norm(model)
    results["checks"].append({
        "name": "gradient_norm",
        "passed": 0 < grad_norm < 1000,
        "value": grad_norm,
    })
    
    # 6. Tied embeddings
    logger.info("Checking tied embeddings...")
    tied = (
        model.shared.weight.data_ptr() == 
        model.encoder.embed_tokens.weight.data_ptr() ==
        model.decoder.embed_tokens.weight.data_ptr()
    )
    results["checks"].append({
        "name": "tied_embeddings",
        "passed": tied,
    })
    if not tied:
        results["passed"] = False
    
    # 7. Generation
    logger.info("Testing generation...")
    model.zero_grad()
    model.eval()
    
    try:
        with torch.no_grad():
            encoder_inputs = batch["input_ids"][:1]  # Single example
            generated = model.generate(
                input_ids=encoder_inputs,
                max_new_tokens=20,
                num_beams=1,
            )
        results["checks"].append({
            "name": "generation",
            "passed": True,
            "output_length": generated.shape[1],
        })
    except Exception as e:
        results["checks"].append({
            "name": "generation",
            "passed": False,
            "error": str(e),
        })
    
    # Memory stats
    mem_stats = get_memory_stats()
    results["memory"] = mem_stats
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    results = run_sanity_check(args.model_path, args.device)
    
    # Print results
    print("\n" + "=" * 60)
    print("SANITY CHECK RESULTS")
    print("=" * 60)
    
    for check in results["checks"]:
        status = "✓" if check["passed"] else "✗"
        print(f"  {status} {check['name']}")
        if "error" in check:
            print(f"      Error: {check['error']}")
        if "value" in check:
            print(f"      Value: {check['value']}")
    
    print("=" * 60)
    if results["passed"]:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
    print("=" * 60)
    
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    exit(main())
```

### 9.4 Validation Training Script

```python
# scripts/validation_run.py

"""
Validation training run - 1B tokens to verify convergence.

Usage:
    accelerate launch scripts/validation_run.py \
        --model-path ./initialized-model \
        --output-dir ./validation-output
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.modeling_qwen3_encdec import Qwen3ForSeq2SeqLM
from src.tokenization_qwen3_encdec import Qwen3EncoderDecoderTokenizer
from src.data.ul2_dataset import create_ul2_dataset
from src.data.collator import UL2DataCollator
from src.training.config import FullConfig
from src.training.monitor import TrainingMonitor, compute_gradient_norm
from src.training.execution import PHASE_CONFIGS, TrainingPhase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_validation(
    model_path: str,
    output_dir: str,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    num_steps: int = 5000,
):
    """Run validation training."""
    phase_config = PHASE_CONFIGS[TrainingPhase.VALIDATION]
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
    )
    
    # Load model and tokenizer
    logger.info("Loading model...")
    model = Qwen3ForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(model_path)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_ul2_dataset(
        dataset_name,
        tokenizer,
        streaming=True,
        max_seq_length=2048,
    )
    
    collator = UL2DataCollator(
        tokenizer,
        max_encoder_length=1024,
        max_decoder_length=512,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        num_workers=2,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01,
    )
    
    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    # Training monitor
    monitor = TrainingMonitor()
    
    # Training loop
    logger.info(f"Starting validation run ({num_steps} steps)...")
    model.train()
    
    progress = tqdm(range(num_steps), disable=not accelerator.is_local_main_process)
    data_iter = iter(dataloader)
    
    warmup_steps = 100
    total_tokens = 0
    
    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            # Compute metrics
            grad_norm = compute_gradient_norm(accelerator.unwrap_model(model))
            batch_tokens = batch["input_ids"].numel() + batch["decoder_input_ids"].numel()
            total_tokens += batch_tokens * accelerator.num_processes
            
            # Update monitor
            monitor.update(
                step=step,
                loss=loss.item(),
                grad_norm=grad_norm,
                learning_rate=optimizer.param_groups[0]["lr"],
                tokens_per_second=batch_tokens / 0.5,  # Rough estimate
                is_warmup=(step < warmup_steps),
            )
            
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm:.2f}",
            )
    
    # Validate results
    validation_results = monitor.validate_phase(phase_config)
    
    # Print summary
    if accelerator.is_main_process:
        summary = monitor.get_summary()
        
        print("\n" + "=" * 60)
        print("VALIDATION RUN SUMMARY")
        print("=" * 60)
        print(f"Steps completed: {summary['steps']}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Initial loss: {summary['loss']['initial']:.4f}")
        print(f"Final loss: {summary['loss']['current']:.4f}")
        print(f"Loss decrease: {summary['loss']['decrease']:.4f}")
        print(f"Max grad norm: {summary['grad_norm']['max']:.4f}")
        print(f"Alerts: {summary['alerts']}")
        print("=" * 60)
        
        print("\nValidation Checks:")
        for check in validation_results["checks"]:
            status = "✓" if check["passed"] else "✗"
            print(f"  {status} {check['name']}: {check['message']}")
        
        if validation_results["passed"]:
            print("\n✓ VALIDATION PASSED - Ready for full training")
        else:
            print("\n✗ VALIDATION FAILED - Review issues before proceeding")
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path / "validation_results.json", "w") as f:
            json.dump({
                "summary": summary,
                "validation": validation_results,
            }, f, indent=2)
        
        # Save model if passed
        if validation_results["passed"]:
            accelerator.unwrap_model(model).save_pretrained(output_path / "model")
            tokenizer.save_pretrained(output_path / "model")
    
    return validation_results["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--num-steps", type=int, default=5000)
    args = parser.parse_args()
    
    success = run_validation(
        args.model_path,
        args.output_dir,
        args.dataset,
        args.num_steps,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
```

### 9.5 Full Training Launch

```bash
#!/bin/bash
# scripts/run_full_training.sh

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-./initialized-model}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/full-training-$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"

# Training phases
PHASE="${PHASE:-full}"  # sanity_check, validation, medium, full

echo "=========================================="
echo "Qwen3 Encoder-Decoder Training Pipeline"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Phase: $PHASE"
echo "=========================================="

# Step 1: Sanity Check
if [ "$PHASE" == "sanity_check" ] || [ "$PHASE" == "full" ]; then
    echo ""
    echo "Step 1: Running sanity check..."
    python scripts/sanity_check.py --model-path "$MODEL_PATH"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Sanity check failed!"
        exit 1
    fi
    echo "✓ Sanity check passed"
fi

# Step 2: Validation Run
if [ "$PHASE" == "validation" ] || [ "$PHASE" == "full" ]; then
    echo ""
    echo "Step 2: Running validation (1B tokens)..."
    accelerate launch \
        --config_file configs/accelerate_fsdp2.yaml \
        scripts/validation_run.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR/validation" \
        --dataset "$DATASET" \
        --num-steps 5000
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Validation run failed!"
        exit 1
    fi
    echo "✓ Validation passed"
fi

# Step 3: Full Training
if [ "$PHASE" == "medium" ] || [ "$PHASE" == "full" ]; then
    echo ""
    echo "Step 3: Running full training..."
    
    # Use validation checkpoint if available
    if [ -d "$OUTPUT_DIR/validation/model" ]; then
        TRAIN_MODEL="$OUTPUT_DIR/validation/model"
    else
        TRAIN_MODEL="$MODEL_PATH"
    fi
    
    accelerate launch \
        --config_file configs/accelerate_fsdp2.yaml \
        scripts/train.py \
        --config configs/training_config.yaml \
        --output_dir "$OUTPUT_DIR/training"
    
    echo "✓ Training complete"
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
```

---

## Unit Tests

```python
# tests/test_training_execution.py

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

from src.training.execution import (
    TrainingPhase,
    PhaseConfig,
    PHASE_CONFIGS,
    estimate_steps_for_tokens,
    estimate_training_time,
)
from src.training.monitor import (
    MetricWindow,
    TrainingMonitor,
    compute_gradient_norm,
)


class TestPhaseConfig:
    """Tests for training phase configuration."""
    
    def test_phase_configs_exist(self):
        """Test all phases have configurations."""
        for phase in TrainingPhase:
            assert phase in PHASE_CONFIGS
    
    def test_phase_config_values(self):
        """Test phase config values are reasonable."""
        for phase, config in PHASE_CONFIGS.items():
            assert config.num_tokens > 0
            assert config.max_loss_after_warmup > 0
            assert config.num_train_steps > 0
    
    def test_validation_phase_is_smaller_than_full(self):
        """Test validation uses fewer tokens than full."""
        val = PHASE_CONFIGS[TrainingPhase.VALIDATION]
        full = PHASE_CONFIGS[TrainingPhase.FULL]
        
        assert val.num_tokens < full.num_tokens
        assert val.num_train_steps < full.num_train_steps


class TestStepEstimation:
    """Tests for step and time estimation."""
    
    def test_estimate_steps_basic(self):
        """Test basic step estimation."""
        steps = estimate_steps_for_tokens(
            num_tokens=1_000_000,
            batch_size=4,
            seq_length=1000,
            gradient_accumulation_steps=2,
            num_gpus=1,
        )
        
        # 1M tokens / (4 * 1000 * 2 * 1) = 125 steps
        assert steps == 125
    
    def test_estimate_steps_multi_gpu(self):
        """Test step estimation scales with GPUs."""
        single_gpu = estimate_steps_for_tokens(
            num_tokens=1_000_000,
            batch_size=4,
            seq_length=1000,
            gradient_accumulation_steps=2,
            num_gpus=1,
        )
        
        multi_gpu = estimate_steps_for_tokens(
            num_tokens=1_000_000,
            batch_size=4,
            seq_length=1000,
            gradient_accumulation_steps=2,
            num_gpus=8,
        )
        
        assert multi_gpu == single_gpu // 8
    
    def test_estimate_training_time(self):
        """Test time estimation."""
        time_est = estimate_training_time(
            num_steps=1000,
            step_time_seconds=0.5,
        )
        
        assert time_est["seconds"] == 500
        assert abs(time_est["minutes"] - 500/60) < 0.01


class TestMetricWindow:
    """Tests for MetricWindow."""
    
    def test_window_basic(self):
        """Test basic window operations."""
        window = MetricWindow(window_size=5)
        
        for i in range(5):
            window.add(float(i))
        
        assert window.mean == 2.0
        assert window.min == 0.0
        assert window.max == 4.0
    
    def test_window_sliding(self):
        """Test window slides correctly."""
        window = MetricWindow(window_size=3)
        
        for i in range(10):
            window.add(float(i))
        
        # Should only have last 3 values: 7, 8, 9
        assert window.mean == 8.0
        assert len(window.values) == 3


class TestTrainingMonitor:
    """Tests for TrainingMonitor."""
    
    def test_monitor_tracks_loss(self):
        """Test monitor tracks loss correctly."""
        monitor = TrainingMonitor()
        
        for i in range(100):
            monitor.update(
                step=i,
                loss=10.0 - i * 0.05,  # Decreasing loss
                grad_norm=1.0,
                learning_rate=1e-4,
                tokens_per_second=1000,
            )
        
        assert monitor.initial_loss == 10.0
        assert monitor.loss_window.mean < 10.0
    
    def test_monitor_detects_loss_spike(self):
        """Test monitor detects loss spikes."""
        monitor = TrainingMonitor(loss_spike_threshold=2.0)
        
        # Normal training
        for i in range(20):
            monitor.update(i, loss=5.0, grad_norm=1.0, learning_rate=1e-4, tokens_per_second=1000)
        
        # Spike
        monitor.update(20, loss=20.0, grad_norm=1.0, learning_rate=1e-4, tokens_per_second=1000)
        
        assert len(monitor.alerts) > 0
        assert any(a["type"] == "loss_spike" for a in monitor.alerts)
    
    def test_monitor_detects_grad_explosion(self):
        """Test monitor detects gradient explosion."""
        monitor = TrainingMonitor(grad_norm_threshold=10.0)
        
        monitor.update(0, loss=5.0, grad_norm=100.0, learning_rate=1e-4, tokens_per_second=1000)
        
        assert len(monitor.alerts) > 0
        assert any(a["type"] == "grad_explosion" for a in monitor.alerts)
    
    def test_monitor_validate_phase(self):
        """Test phase validation."""
        monitor = TrainingMonitor()
        
        # Simulate good training
        monitor.initial_loss = 10.0
        monitor.warmup_complete_loss = 5.0
        for _ in range(100):
            monitor.loss_window.add(3.0)
        
        phase_config = PhaseConfig(
            phase=TrainingPhase.VALIDATION,
            num_tokens=1_000_000,
            description="Test",
            max_loss_after_warmup=8.0,
            min_loss_decrease=0.5,
        )
        
        results = monitor.validate_phase(phase_config)
        
        assert results["passed"]
        assert len(results["checks"]) >= 2


class TestGradientNorm:
    """Tests for gradient norm computation."""
    
    def test_compute_gradient_norm(self):
        """Test gradient norm computation."""
        model = torch.nn.Linear(10, 10)
        
        # Forward + backward to get gradients
        x = torch.randn(5, 10)
        loss = model(x).sum()
        loss.backward()
        
        grad_norm = compute_gradient_norm(model)
        
        assert grad_norm > 0
        assert not np.isnan(grad_norm)
        assert not np.isinf(grad_norm)
    
    def test_compute_gradient_norm_no_grads(self):
        """Test gradient norm with no gradients."""
        model = torch.nn.Linear(10, 10)
        
        grad_norm = compute_gradient_norm(model)
        
        assert grad_norm == 0.0
```

---

## Acceptance Criteria

1. **Sanity Check**
   - [ ] Verifies model loads correctly
   - [ ] Confirms forward/backward pass works
   - [ ] Checks gradients flow to encoder
   - [ ] Validates tied embeddings
   - [ ] Tests generation capability

2. **Training Monitor**
   - [ ] Tracks loss, gradient norm, throughput
   - [ ] Detects loss spikes
   - [ ] Detects gradient explosion
   - [ ] Validates against phase criteria

3. **Validation Run**
   - [ ] Completes 1B token training
   - [ ] Loss decreases as expected
   - [ ] No critical alerts
   - [ ] Saves checkpoint on success

4. **Full Training Pipeline**
   - [ ] Phases execute in sequence
   - [ ] Early stopping on failure
   - [ ] Proper checkpointing
   - [ ] Comprehensive logging

---

## Dependencies

- **Stories 01-08**: All prior infrastructure

---

## Estimated Effort

- Sanity check script: 0.5 days
- Training monitor: 1 day
- Validation run: 0.5 days
- Full pipeline: 0.5 days
- Testing: 0.5 days
- **Total: 2-3 days**

---

## Developer Notes

1. **Validate Before Scale**: Always run sanity check and validation before committing to full training.

2. **Monitor Alerts**: Take alerts seriously - loss spikes often indicate data issues or learning rate problems.

3. **Checkpoint Frequently**: Full training can fail at any point. Frequent checkpoints prevent losing progress.

4. **Loss Expectations**: Initial loss should be ~log(vocab_size) ≈ 12. After training, expect 2-4.

5. **Throughput Targets**: Aim for >10K tokens/second/GPU with proper optimization.
