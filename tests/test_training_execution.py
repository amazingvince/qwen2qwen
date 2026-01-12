"""Tests for training execution and monitoring."""

import numpy as np
import torch

from src.training.execution import (
    PHASE_CONFIGS,
    PhaseConfig,
    TrainingPhase,
    estimate_steps_for_tokens,
    estimate_training_time,
)
from src.training.monitor import MetricWindow, TrainingMonitor, compute_gradient_norm

# =============================================================================
# Test Phase Configuration
# =============================================================================


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

    def test_phase_configs_progressive(self):
        """Test phases scale progressively."""
        sanity = PHASE_CONFIGS[TrainingPhase.SANITY_CHECK]
        validation = PHASE_CONFIGS[TrainingPhase.VALIDATION]
        medium = PHASE_CONFIGS[TrainingPhase.MEDIUM]
        full = PHASE_CONFIGS[TrainingPhase.FULL]

        # Tokens should increase
        assert sanity.num_tokens < validation.num_tokens
        assert validation.num_tokens < medium.num_tokens
        assert medium.num_tokens < full.num_tokens

        # Steps should increase
        assert sanity.num_train_steps < validation.num_train_steps
        assert validation.num_train_steps < medium.num_train_steps


# =============================================================================
# Test Step Estimation
# =============================================================================


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

    def test_estimate_steps_gradient_accumulation(self):
        """Test gradient accumulation affects step count."""
        no_accum = estimate_steps_for_tokens(
            num_tokens=1_000_000,
            batch_size=4,
            seq_length=1000,
            gradient_accumulation_steps=1,
            num_gpus=1,
        )

        with_accum = estimate_steps_for_tokens(
            num_tokens=1_000_000,
            batch_size=4,
            seq_length=1000,
            gradient_accumulation_steps=4,
            num_gpus=1,
        )

        assert with_accum == no_accum // 4

    def test_estimate_training_time(self):
        """Test time estimation."""
        time_est = estimate_training_time(
            num_steps=1000,
            step_time_seconds=0.5,
        )

        assert time_est["seconds"] == 500
        assert abs(time_est["minutes"] - 500 / 60) < 0.01
        assert abs(time_est["hours"] - 500 / 3600) < 0.001
        assert abs(time_est["days"] - 500 / 86400) < 0.0001

    def test_estimate_training_time_longer(self):
        """Test time estimation for longer runs."""
        time_est = estimate_training_time(
            num_steps=100000,
            step_time_seconds=1.0,
        )

        # 100000 steps * 1 second = 100000 seconds
        assert time_est["seconds"] == 100000
        assert abs(time_est["days"] - 100000 / 86400) < 0.001


# =============================================================================
# Test MetricWindow
# =============================================================================


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

    def test_window_std(self):
        """Test standard deviation calculation."""
        window = MetricWindow(window_size=5)

        # Add same value - std should be 0
        for _ in range(5):
            window.add(5.0)

        assert window.std == 0.0

        # Add different values
        window2 = MetricWindow(window_size=3)
        window2.add(0.0)
        window2.add(5.0)
        window2.add(10.0)

        assert window2.std > 0

    def test_window_empty(self):
        """Test empty window returns sensible defaults."""
        window = MetricWindow(window_size=5)

        assert window.mean == 0.0
        assert window.std == 0.0
        assert window.min == float("inf")
        assert window.max == float("-inf")


# =============================================================================
# Test TrainingMonitor
# =============================================================================


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
            monitor.update(
                i, loss=5.0, grad_norm=1.0, learning_rate=1e-4, tokens_per_second=1000
            )

        # Spike
        monitor.update(
            20, loss=20.0, grad_norm=1.0, learning_rate=1e-4, tokens_per_second=1000
        )

        assert len(monitor.alerts) > 0
        assert any(a["type"] == "loss_spike" for a in monitor.alerts)

    def test_monitor_detects_grad_explosion(self):
        """Test monitor detects gradient explosion."""
        monitor = TrainingMonitor(grad_norm_threshold=10.0)

        monitor.update(
            0, loss=5.0, grad_norm=100.0, learning_rate=1e-4, tokens_per_second=1000
        )

        assert len(monitor.alerts) > 0
        assert any(a["type"] == "grad_explosion" for a in monitor.alerts)

    def test_monitor_warmup_loss_tracking(self):
        """Test warmup loss tracking."""
        monitor = TrainingMonitor()

        # Warmup phase
        for i in range(10):
            monitor.update(
                step=i,
                loss=10.0 - i * 0.1,
                grad_norm=1.0,
                learning_rate=1e-4,
                tokens_per_second=1000,
                is_warmup=True,
            )

        # No warmup complete loss yet (still in warmup)
        assert monitor.warmup_complete_loss is None

        # Post-warmup
        monitor.update(
            step=10,
            loss=8.5,
            grad_norm=1.0,
            learning_rate=1e-4,
            tokens_per_second=1000,
            is_warmup=False,
        )

        assert monitor.warmup_complete_loss == 8.5

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

    def test_monitor_validate_phase_fails_high_loss(self):
        """Test phase validation fails on high loss."""
        monitor = TrainingMonitor()

        monitor.initial_loss = 10.0
        monitor.warmup_complete_loss = 15.0  # Too high
        for _ in range(100):
            monitor.loss_window.add(12.0)

        phase_config = PhaseConfig(
            phase=TrainingPhase.VALIDATION,
            num_tokens=1_000_000,
            description="Test",
            max_loss_after_warmup=8.0,
            min_loss_decrease=0.5,
        )

        results = monitor.validate_phase(phase_config)

        assert not results["passed"]

    def test_monitor_get_summary(self):
        """Test summary generation."""
        monitor = TrainingMonitor()

        for i in range(50):
            monitor.update(
                step=i,
                loss=10.0 - i * 0.1,
                grad_norm=1.0 + i * 0.01,
                learning_rate=1e-4,
                tokens_per_second=1000,
            )

        summary = monitor.get_summary()

        assert "loss" in summary
        assert "grad_norm" in summary
        assert "throughput" in summary
        assert "alerts" in summary
        assert "steps" in summary

        assert summary["steps"] == 50
        assert summary["loss"]["initial"] == 10.0


# =============================================================================
# Test Gradient Norm
# =============================================================================


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

    def test_compute_gradient_norm_partial_grads(self):
        """Test gradient norm with partial gradients."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
        )

        # Only compute grads for first layer
        x = torch.randn(5, 10)
        intermediate = model[0](x)
        loss = intermediate.sum()
        loss.backward()

        grad_norm = compute_gradient_norm(model)

        # Should only count first layer's gradients
        assert grad_norm > 0

    def test_compute_gradient_norm_multi_layer(self):
        """Test gradient norm across multiple layers."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
        )

        x = torch.randn(5, 10)
        loss = model(x).sum()
        loss.backward()

        grad_norm = compute_gradient_norm(model)

        assert grad_norm > 0
        assert not np.isnan(grad_norm)
