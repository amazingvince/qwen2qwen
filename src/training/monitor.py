"""
Training monitoring and validation utilities.

Provides tools for tracking training metrics, detecting issues,
and validating against phase criteria.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MetricWindow:
    """Sliding window for metric tracking."""

    window_size: int = 100
    _values: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        # Ensure deque has correct maxlen
        self._values = deque(maxlen=self.window_size)

    def add(self, value: float) -> None:
        """Add a value to the window."""
        self._values.append(value)

    @property
    def values(self) -> Deque[float]:
        """Get the underlying deque."""
        return self._values

    @property
    def mean(self) -> float:
        """Get the mean of values in the window."""
        if not self._values:
            return 0.0
        return float(np.mean(list(self._values)))

    @property
    def std(self) -> float:
        """Get the standard deviation of values in the window."""
        if len(self._values) < 2:
            return 0.0
        return float(np.std(list(self._values)))

    @property
    def min(self) -> float:
        """Get the minimum value in the window."""
        if not self._values:
            return float("inf")
        return min(self._values)

    @property
    def max(self) -> float:
        """Get the maximum value in the window."""
        if not self._values:
            return float("-inf")
        return max(self._values)


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
    ) -> None:
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

    def _check_loss_spike(self, step: int, loss: float) -> None:
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
                "message": (
                    f"Loss spike detected: {loss:.4f} > "
                    f"{recent_mean * self.loss_spike_threshold:.4f}"
                ),
            }
            self.alerts.append(alert)
            logger.warning(alert["message"])

    def _check_grad_norm(self, step: int, grad_norm: float) -> None:
        """Check for gradient explosion."""
        if grad_norm > self.grad_norm_threshold:
            alert = {
                "type": "grad_explosion",
                "step": step,
                "value": grad_norm,
                "threshold": self.grad_norm_threshold,
                "message": (
                    f"High gradient norm: {grad_norm:.4f} > {self.grad_norm_threshold}"
                ),
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
                "decrease": (
                    (self.initial_loss - self.loss_window.mean)
                    if self.initial_loss
                    else 0
                ),
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

    def validate_phase(self, phase_config: Any) -> Dict[str, Any]:
        """
        Validate training meets phase criteria.

        Args:
            phase_config: PhaseConfig with validation thresholds.

        Returns:
            Validation results with pass/fail status
        """
        results: Dict[str, Any] = {
            "passed": True,
            "checks": [],
        }

        # Check loss after warmup
        if self.warmup_complete_loss is not None:
            if self.warmup_complete_loss > phase_config.max_loss_after_warmup:
                results["passed"] = False
                results["checks"].append(
                    {
                        "name": "loss_after_warmup",
                        "passed": False,
                        "message": (
                            f"Loss {self.warmup_complete_loss:.4f} exceeds "
                            f"max {phase_config.max_loss_after_warmup}"
                        ),
                    }
                )
            else:
                results["checks"].append(
                    {
                        "name": "loss_after_warmup",
                        "passed": True,
                        "message": f"Loss {self.warmup_complete_loss:.4f} within limit",
                    }
                )

        # Check loss decrease
        if self.initial_loss and self.loss_window.mean:
            loss_decrease = self.initial_loss - self.loss_window.mean
            if loss_decrease < phase_config.min_loss_decrease:
                results["passed"] = False
                results["checks"].append(
                    {
                        "name": "loss_decrease",
                        "passed": False,
                        "message": (
                            f"Loss decrease {loss_decrease:.4f} below "
                            f"min {phase_config.min_loss_decrease}"
                        ),
                    }
                )
            else:
                results["checks"].append(
                    {
                        "name": "loss_decrease",
                        "passed": True,
                        "message": f"Loss decreased by {loss_decrease:.4f}",
                    }
                )

        # Check for alerts
        critical_alerts = [a for a in self.alerts if a["type"] == "loss_spike"]
        if len(critical_alerts) > 5:
            results["passed"] = False
            results["checks"].append(
                {
                    "name": "loss_spikes",
                    "passed": False,
                    "message": f"Too many loss spikes: {len(critical_alerts)}",
                }
            )

        return results


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute total gradient norm across all parameters.

    Args:
        model: PyTorch model to compute gradient norm for.

    Returns:
        L2 norm of all gradients.
    """
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm**0.5
