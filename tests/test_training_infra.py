"""Tests for training infrastructure."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.training import (
    DataConfig,
    FullConfig,
    InfraConfig,
    ModelConfig,
    TrainingConfig,
    TrainingState,
    clear_memory,
    estimate_model_memory,
    get_memory_stats,
)


# =============================================================================
# Test Configurations
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test ModelConfig default values."""
        config = ModelConfig()

        assert config.model_name_or_path == "Qwen/Qwen3-0.6B"
        assert config.num_sentinel_tokens == 100
        assert config.use_flash_attention is True

    def test_tokenizer_defaults_to_model(self):
        """Test tokenizer_name_or_path defaults to model_name_or_path."""
        config = ModelConfig(model_name_or_path="custom/model")

        assert config.tokenizer_name_or_path == "custom/model"

    def test_explicit_tokenizer(self):
        """Test explicit tokenizer_name_or_path."""
        config = ModelConfig(
            model_name_or_path="custom/model",
            tokenizer_name_or_path="different/tokenizer",
        )

        assert config.tokenizer_name_or_path == "different/tokenizer"


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test DataConfig default values."""
        config = DataConfig()

        assert config.dataset_name == "HuggingFaceFW/fineweb-edu"
        assert config.streaming is True
        assert config.max_seq_length == 8192
        assert config.max_encoder_length == 4096
        assert config.max_decoder_length == 2048

    def test_ul2_weights(self):
        """Test UL2 task weights are T5Gemma 2 defaults."""
        config = DataConfig()

        assert config.ul2_task_weights == [1, 1, 1, 1, 4]


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()

        assert config.learning_rate == 2e-4
        assert config.max_grad_norm == 1.0
        assert config.bf16 is True
        assert config.fp16 is False
        assert config.warmup_steps == 1000

    def test_effective_batch_size(self):
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


class TestInfraConfig:
    """Tests for InfraConfig."""

    def test_default_values(self):
        """Test InfraConfig default values."""
        config = InfraConfig()

        assert config.distributed_type == "fsdp"
        assert config.fsdp_sharding_strategy == "FULL_SHARD"
        assert config.fsdp_activation_checkpointing is True
        assert config.wandb_project == "qwen3-encoder-decoder"


class TestFullConfig:
    """Tests for FullConfig."""

    def test_default_subconfigs(self):
        """Test FullConfig creates all subconfigs."""
        config = FullConfig()

        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.infra, InfraConfig)

    def test_to_yaml_and_from_yaml(self):
        """Test saving and loading config to YAML."""
        config = FullConfig(
            model=ModelConfig(num_sentinel_tokens=50),
            training=TrainingConfig(learning_rate=1e-4),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(path))

            assert path.exists()

            # Load and verify
            loaded = FullConfig.from_yaml(str(path))
            assert loaded.model.num_sentinel_tokens == 50
            assert loaded.training.learning_rate == 1e-4

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = FullConfig()
        config_dict = config.to_dict()

        assert "model" in config_dict
        assert "data" in config_dict
        assert "training" in config_dict
        assert "infra" in config_dict

        assert config_dict["model"]["num_sentinel_tokens"] == 100


# =============================================================================
# Test Memory Utilities
# =============================================================================


class TestMemoryUtils:
    """Tests for memory utilities."""

    def test_get_memory_stats_structure(self):
        """Test memory stats returns correct structure."""
        stats = get_memory_stats()

        if torch.cuda.is_available():
            assert "allocated_gb" in stats
            assert "reserved_gb" in stats
            assert "max_allocated_gb" in stats
        else:
            assert stats.get("gpu_available") is False

    def test_estimate_model_memory(self):
        """Test memory estimation returns reasonable values."""
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
        assert "gradient_gb" in estimates
        assert "activation_gb" in estimates
        assert "total_gb" in estimates

        # Sanity checks
        assert estimates["model_gb"] > 0
        # Optimizer states should be larger (2 FP32 states)
        assert estimates["optimizer_gb"] > estimates["model_gb"]
        assert estimates["total_gb"] > estimates["model_gb"]

    def test_estimate_checkpointing_reduces_memory(self):
        """Test that gradient checkpointing reduces activation memory."""
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

    def test_clear_memory_no_error(self):
        """Test clear_memory runs without error."""
        # Should not raise
        clear_memory()


# =============================================================================
# Test Trainer Components
# =============================================================================


class TestTrainingState:
    """Tests for TrainingState."""

    def test_default_values(self):
        """Test TrainingState default values."""
        state = TrainingState()

        assert state.global_step == 0
        assert state.epoch == 0
        assert state.best_eval_loss == float("inf")
        assert state.total_tokens_seen == 0

    def test_save_and_load(self):
        """Test saving and loading training state."""
        state = TrainingState(
            global_step=1000,
            epoch=2,
            best_eval_loss=0.5,
            total_tokens_seen=1_000_000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.pt"

            torch.save(
                {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "best_eval_loss": state.best_eval_loss,
                    "total_tokens_seen": state.total_tokens_seen,
                },
                path,
            )

            loaded = torch.load(path)

            assert loaded["global_step"] == 1000
            assert loaded["epoch"] == 2
            assert loaded["best_eval_loss"] == 0.5
            assert loaded["total_tokens_seen"] == 1_000_000


class TestOptimizerParamGroups:
    """Tests for optimizer parameter group separation."""

    def test_separate_decay_params(self):
        """Test that bias and norm params are separated."""
        # Create model with named modules (like real transformer)
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.layernorm = torch.nn.LayerNorm(10)
                self.linear2 = torch.nn.Linear(10, 10)

        model = SimpleModel()

        # Separate params like trainer does
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # 2 Linear weights should decay
        assert len(decay_params) == 2
        # 2 Linear biases + 2 LayerNorm params (weight + bias)
        assert len(no_decay_params) == 4


class TestScheduler:
    """Tests for learning rate scheduler."""

    def test_warmup_starts_low(self):
        """Test scheduler starts with low learning rate during warmup."""
        from torch.optim.lr_scheduler import LinearLR

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=100,
        )

        # Initial LR should be low
        assert warmup_scheduler.get_last_lr()[0] < 1e-4

    def test_warmup_reaches_peak(self):
        """Test scheduler reaches peak LR after warmup."""
        from torch.optim.lr_scheduler import LinearLR

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=100,
        )

        # Step through warmup
        for _ in range(100):
            warmup_scheduler.step()

        # Should be at full LR
        assert abs(warmup_scheduler.get_last_lr()[0] - 1e-4) < 1e-8


class TestCheckpointAveraging:
    """Tests for checkpoint averaging."""

    def test_averaging_math(self):
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

    def test_averaging_preserves_dtype(self):
        """Test averaging can convert back to target dtype."""
        ckpts = [
            {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
            {"weight": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)},
        ]

        # Average in float32
        avg_state = {}
        for ckpt in ckpts:
            for key, value in ckpt.items():
                if key not in avg_state:
                    avg_state[key] = value.float() / len(ckpts)
                else:
                    avg_state[key] += value.float() / len(ckpts)

        # Convert back
        for key in avg_state:
            avg_state[key] = avg_state[key].to(torch.bfloat16)

        assert avg_state["weight"].dtype == torch.bfloat16


# =============================================================================
# Test Config Files
# =============================================================================


class TestConfigFiles:
    """Tests for config file loading."""

    def test_load_training_config(self):
        """Test loading the default training config if it exists."""
        config_path = Path("configs/training_config.yaml")

        if config_path.exists():
            config = FullConfig.from_yaml(str(config_path))

            # Verify some expected values
            assert config.model.num_sentinel_tokens == 100
            assert config.training.bf16 is True
