"""Tests for UL2_5-backed UL2 data pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.data import (UL2DataCollator, create_collator_from_config,
                      t5gemma2_config, ul2_recommended_config,
                      ul2_recommended_with_curriculum_config)


@dataclass
class MockTokenizer:
    eos_token_id: int = 1
    pad_token_id: int = 0
    bos_token_id: int = 2
    unk_token_id: int = 3
    original_vocab_size: int = 1000
    num_sentinel_tokens: int = 100

    def convert_tokens_to_ids(self, token: str) -> int:
        if token.startswith("<extra_id_") and token.endswith(">"):
            idx = int(token[len("<extra_id_") : -1])
            return self.original_vocab_size + idx
        if token in {"[R]", "[X]", "[S]", "[I]"}:
            return {"[R]": 200, "[X]": 201, "[S]": 202, "[I]": 203}[token]
        return self.unk_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [100 + (ord(c) % 50) for c in text]


class TestT5Gemma2Config:
    def test_default_weights_normalized(self):
        cfg = t5gemma2_config()
        assert len(cfg.denoisers) == 5
        assert len(cfg.weights) == 5
        assert abs(sum(cfg.weights) - 1.0) < 1e-6

        expected = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 4 / 8]
        for got, exp in zip(cfg.weights, expected):
            assert abs(got - exp) < 1e-6

    def test_override_weights(self):
        cfg = t5gemma2_config(weights=[3, 1, 1, 1, 2])
        assert len(cfg.weights) == 5
        assert abs(sum(cfg.weights) - 1.0) < 1e-6


class TestUL2DataCollator:
    def test_collator_outputs_expected_keys(self):
        torch.manual_seed(0)
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(
            tokenizer,
            config=t5gemma2_config(enable_length_adaptive=False),
            max_length=64,
            max_labels_length=32,
            collate_on_cpu=True,
        )

        examples = [{"input_ids": torch.randint(10, 900, (48,))} for _ in range(4)]
        batch = collator(examples)

        assert set(batch.keys()) >= {
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        }
        assert batch["input_ids"].shape[0] == 4
        assert batch["labels"].shape[0] == 4
        assert batch["decoder_input_ids"].shape == batch["labels"].shape
        assert batch["decoder_attention_mask"].shape == batch["labels"].shape
        assert batch["decoder_input_ids"][:, 0].eq(tokenizer.bos_token_id).all()

    def test_sentinels_are_in_sentinel_range(self):
        torch.manual_seed(0)
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(
            tokenizer,
            config=t5gemma2_config(enable_length_adaptive=False),
            max_length=64,
            max_labels_length=32,
            collate_on_cpu=True,
        )

        examples = [{"input_ids": torch.randint(10, 900, (48,))} for _ in range(2)]
        batch = collator(examples)

        low = tokenizer.original_vocab_size
        high = tokenizer.original_vocab_size + tokenizer.num_sentinel_tokens
        has_sentinel = ((batch["input_ids"] >= low) & (batch["input_ids"] < high)).any()
        assert bool(has_sentinel)

    def test_progress_property_roundtrip(self):
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(tokenizer, config=t5gemma2_config())
        collator.progress = 0.5
        assert 0.49 < collator.progress < 0.51


@pytest.mark.parametrize("bad_weights", [[1, 1, 1], [], [0, 0, 0, 0, 0]])
def test_invalid_t5gemma2_weights_rejected(bad_weights):
    if len(bad_weights) != 5 or sum(bad_weights) <= 0:
        with pytest.raises(Exception):
            t5gemma2_config(weights=bad_weights)  # type: ignore[arg-type]


class TestUL2RecommendedConfigs:
    """Tests for the new recommended config functions."""

    def test_recommended_config_returns_valid_config(self):
        cfg = ul2_recommended_config()
        assert cfg is not None
        assert hasattr(cfg, "denoisers")
        assert hasattr(cfg, "weights")
        assert len(cfg.denoisers) > 0
        assert len(cfg.weights) > 0

    def test_recommended_config_with_unpad_options(self):
        cfg = ul2_recommended_config(
            enable_unpad_encoder=True,
            enable_unpad_decoder=True,
        )
        assert cfg.enable_unpad_encoder is True
        assert cfg.enable_unpad_decoder is True

    def test_recommended_with_curriculum_returns_valid_config(self):
        cfg = ul2_recommended_with_curriculum_config()
        assert cfg is not None
        assert hasattr(cfg, "denoisers")
        assert hasattr(cfg, "weights")
        # Curriculum configs should have curriculum_start/end defined
        assert cfg.curriculum_start is not None or cfg.curriculum_end is not None


@dataclass
class MockDataConfig:
    """Mock DataConfig for testing create_collator_from_config."""

    max_encoder_length: int = 512
    max_decoder_length: int = 128
    dataloader_collate_on_cpu: bool = True
    ul2_curriculum_start: list = None
    ul2_curriculum_end: list = None
    ul2_unpad_encoder: bool = False
    ul2_unpad_decoder: bool = False
    ul2_length_adaptive: bool = False
    ul2_boundary_snapping: bool = False


class TestCreateCollatorFromConfig:
    """Tests for the factory function."""

    def test_factory_creates_collator_with_defaults(self):
        tokenizer = MockTokenizer()
        config = MockDataConfig()
        collator = create_collator_from_config(tokenizer, config)

        assert isinstance(collator, UL2DataCollator)
        assert collator.max_length == config.max_encoder_length
        assert collator.max_labels_length == config.max_decoder_length

    def test_factory_uses_curriculum_config_when_curriculum_set(self):
        tokenizer = MockTokenizer()
        config = MockDataConfig(
            ul2_curriculum_start=[1.0, 1.0, 1.0, 1.0, 1.0],
            ul2_curriculum_end=[0.0, 0.0, 0.0, 0.0, 5.0],
        )
        collator = create_collator_from_config(tokenizer, config)

        assert isinstance(collator, UL2DataCollator)
        # Curriculum config should have curriculum_start/end
        assert (
            collator.config.curriculum_start is not None
            or collator.config.curriculum_end is not None
        )

    def test_factory_passes_unpad_options(self):
        tokenizer = MockTokenizer()
        config = MockDataConfig(
            ul2_unpad_encoder=True,
            ul2_unpad_decoder=True,
        )
        collator = create_collator_from_config(tokenizer, config)

        assert collator.config.enable_unpad_encoder is True
        assert collator.config.enable_unpad_decoder is True

    def test_factory_passes_length_adaptive_option(self):
        """Regression test: ul2_length_adaptive must be honored."""
        tokenizer = MockTokenizer()
        config = MockDataConfig(ul2_length_adaptive=True)
        collator = create_collator_from_config(tokenizer, config)

        assert collator.config.enable_length_adaptive is True

    def test_factory_passes_boundary_snapping_option(self):
        """Regression test: ul2_boundary_snapping must be honored."""
        tokenizer = MockTokenizer()
        config = MockDataConfig(ul2_boundary_snapping=True)
        collator = create_collator_from_config(tokenizer, config)

        assert collator.config.enable_boundary_snapping is True

    def test_factory_passes_curriculum_arrays(self):
        """Regression test: curriculum_start/end must be passed, not just detected."""
        tokenizer = MockTokenizer()
        start = [0.1, 0.2, 0.3, 0.4, 0.5]
        end = [0.5, 0.4, 0.3, 0.2, 0.1]
        config = MockDataConfig(
            ul2_curriculum_start=start,
            ul2_curriculum_end=end,
        )
        collator = create_collator_from_config(tokenizer, config)

        # The actual values should be passed through, not just used for detection
        assert collator.config.curriculum_start == start
        assert collator.config.curriculum_end == end

    def test_factory_passes_all_ul2_options_together(self):
        """Regression test: all UL2 options must work together."""
        tokenizer = MockTokenizer()
        config = MockDataConfig(
            ul2_unpad_encoder=True,
            ul2_unpad_decoder=True,
            ul2_length_adaptive=True,
            ul2_boundary_snapping=True,
            ul2_curriculum_start=[0.2, 0.2, 0.2, 0.2, 0.2],
            ul2_curriculum_end=[0.1, 0.1, 0.1, 0.1, 0.5],
        )
        collator = create_collator_from_config(tokenizer, config)

        assert collator.config.enable_unpad_encoder is True
        assert collator.config.enable_unpad_decoder is True
        assert collator.config.enable_length_adaptive is True
        assert collator.config.enable_boundary_snapping is True
        assert collator.config.curriculum_start == [0.2, 0.2, 0.2, 0.2, 0.2]
        assert collator.config.curriculum_end == [0.1, 0.1, 0.1, 0.1, 0.5]


class TestUL2ConfigWrappers:
    """Regression tests for the config wrapper functions."""

    def test_recommended_config_passes_length_adaptive(self):
        cfg = ul2_recommended_config(enable_length_adaptive=True)
        assert cfg.enable_length_adaptive is True

    def test_recommended_config_passes_boundary_snapping(self):
        cfg = ul2_recommended_config(enable_boundary_snapping=True)
        assert cfg.enable_boundary_snapping is True

    def test_curriculum_config_passes_length_adaptive(self):
        cfg = ul2_recommended_with_curriculum_config(enable_length_adaptive=True)
        assert cfg.enable_length_adaptive is True

    def test_curriculum_config_passes_boundary_snapping(self):
        cfg = ul2_recommended_with_curriculum_config(enable_boundary_snapping=True)
        assert cfg.enable_boundary_snapping is True

    def test_curriculum_config_passes_curriculum_start(self):
        start = [0.1, 0.2, 0.3]
        cfg = ul2_recommended_with_curriculum_config(curriculum_start=start)
        assert cfg.curriculum_start == start

    def test_curriculum_config_passes_curriculum_end(self):
        end = [0.3, 0.2, 0.1]
        cfg = ul2_recommended_with_curriculum_config(curriculum_end=end)
        assert cfg.curriculum_end == end


class TestCurriculumProgress:
    """Tests for curriculum progress integration."""

    def test_progress_clamps_to_valid_range(self):
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(tokenizer, config=ul2_recommended_config())

        # Values should be clamped 0-1 by the underlying collator
        collator.progress = 0.0
        assert collator.progress >= 0.0

        collator.progress = 1.0
        assert collator.progress <= 1.0

    def test_progress_updates_correctly(self):
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(tokenizer, config=ul2_recommended_config())

        for expected in [0.0, 0.25, 0.5, 0.75, 1.0]:
            collator.progress = expected
            assert abs(collator.progress - expected) < 1e-6


class TestSharedProgress:
    """Regression tests for multiprocessing-safe shared progress."""

    def test_shared_progress_disabled_by_default(self):
        """By default, shared progress is not used."""
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(tokenizer, config=ul2_recommended_config())
        assert collator._shared_progress is None

    def test_shared_progress_enabled_when_requested(self):
        """use_shared_progress=True creates a multiprocessing.Value."""
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(
            tokenizer,
            config=ul2_recommended_config(),
            use_shared_progress=True,
        )
        assert collator._shared_progress is not None

    def test_shared_progress_updates_visible(self):
        """Progress updates via shared memory are immediately visible."""
        tokenizer = MockTokenizer()
        collator = UL2DataCollator(
            tokenizer,
            config=ul2_recommended_config(),
            use_shared_progress=True,
        )

        for expected in [0.0, 0.25, 0.5, 0.75, 1.0]:
            collator.progress = expected
            assert abs(collator.progress - expected) < 1e-6
            # Also verify the underlying shared value
            assert abs(collator._shared_progress.value - expected) < 1e-6

    def test_factory_enables_shared_progress_for_curriculum(self):
        """Regression: factory must enable shared_progress when curriculum is set."""
        tokenizer = MockTokenizer()
        config = MockDataConfig(
            ul2_curriculum_start=[0.2, 0.2, 0.2, 0.2, 0.2],
            ul2_curriculum_end=[0.1, 0.1, 0.1, 0.1, 0.5],
        )
        collator = create_collator_from_config(tokenizer, config)

        # Shared progress must be enabled for curriculum to work with workers
        assert collator._shared_progress is not None

    def test_factory_no_shared_progress_without_curriculum(self):
        """Without curriculum, shared progress is not needed."""
        tokenizer = MockTokenizer()
        config = MockDataConfig()  # No curriculum
        collator = create_collator_from_config(tokenizer, config)

        # No curriculum means no need for shared progress overhead
        assert collator._shared_progress is None
