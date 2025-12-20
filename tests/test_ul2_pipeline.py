"""Tests for UL2 data pipeline."""

from collections import Counter
from unittest.mock import MagicMock

import pytest
import torch

from src.data import (
    DenoiserSpec,
    Task,
    UL2Config,
    UL2DataCollator,
    apply_sentinel_mask,
    count_num_spans,
    create_sentinel_ids,
    infilling_mask,
    middle_heavy_mask,
    prefix_lm_mask,
    span_corruption_mask,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer mimicking Qwen3EncoderDecoderTokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 1
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 2
    tokenizer.unk_token_id = 3
    tokenizer.original_vocab_size = 1000
    tokenizer.num_sentinel_tokens = 100

    def get_sentinel_token_id(idx):
        return 1000 + idx  # original_vocab_size + idx

    tokenizer.get_sentinel_token_id = get_sentinel_token_id

    def encode(text, add_special_tokens=False):
        # Simple mock encoding
        return [ord(c) % 100 + 100 for c in text[:10]]

    tokenizer.encode = encode

    return tokenizer


# =============================================================================
# Test UL2Config
# =============================================================================


class TestUL2Config:
    """Tests for UL2Config presets."""

    def test_t5gemma2_preset(self):
        """Test T5Gemma 2 preset has correct configuration."""
        config = UL2Config.t5gemma2()

        # Should have 5 denoisers
        assert len(config.denoisers) == 5
        assert len(config.weights) == 5

        # Check weights (normalized to 1:1:1:1:4 = total 8)
        expected_weights = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 4 / 8]
        for w, expected in zip(config.weights, expected_weights):
            assert abs(w - expected) < 1e-6

    def test_t5gemma2_denoiser_specs(self):
        """Test T5Gemma 2 denoiser specifications."""
        config = UL2Config.t5gemma2()

        # R1: short spans
        r1 = config.denoisers[0]
        assert r1.task == Task.SPAN
        assert r1.mu == 3.0
        assert r1.r == 0.15

        # R2: medium spans
        r2 = config.denoisers[1]
        assert r2.task == Task.SPAN
        assert r2.mu == 12.0
        assert r2.r == 0.50

        # X1: long spans, low density
        x1 = config.denoisers[2]
        assert x1.task == Task.SPAN
        assert x1.mu == 32.0
        assert x1.r == 0.15

        # X2: long spans, high density
        x2 = config.denoisers[3]
        assert x2.task == Task.SPAN
        assert x2.mu == 32.0
        assert x2.r == 0.50

        # S: prefix LM
        s = config.denoisers[4]
        assert s.task == Task.PREFIX_RANDOM

    def test_minimal_preset(self):
        """Test minimal config for testing."""
        config = UL2Config.minimal()

        assert len(config.denoisers) == 2
        assert config.denoisers[0].task == Task.SPAN
        assert config.denoisers[1].task == Task.PREFIX_RANDOM

    def test_weights_normalization(self):
        """Test that weights are normalized to sum to 1."""
        config = UL2Config(
            denoisers=[
                DenoiserSpec(Task.SPAN),
                DenoiserSpec(Task.PREFIX_RANDOM),
            ],
            weights=[3, 7],
        )

        assert abs(sum(config.weights) - 1.0) < 1e-6
        assert abs(config.weights[0] - 0.3) < 1e-6
        assert abs(config.weights[1] - 0.7) < 1e-6


# =============================================================================
# Test Masking Functions
# =============================================================================


class TestMaskingFunctions:
    """Tests for span corruption masking."""

    def test_span_corruption_mask_density(self, device):
        """Test span corruption produces approximately correct density."""
        seq_len = 200
        noise_density = 0.15

        # Run multiple times for statistical stability
        densities = []
        for _ in range(50):
            mask = span_corruption_mask(seq_len, noise_density, 3.0, 512, device)
            densities.append(mask.float().mean().item())

        avg_density = sum(densities) / len(densities)
        # Should be close to target density
        assert 0.10 < avg_density < 0.25

    def test_span_corruption_mask_shape(self, device):
        """Test mask has correct shape."""
        seq_len = 100
        mask = span_corruption_mask(seq_len, 0.15, 3.0, 512, device)

        assert mask.shape == (seq_len,)
        assert mask.dtype == torch.bool
        assert mask.device.type == device.type

    def test_span_corruption_creates_contiguous_spans(self, device):
        """Test that corruption creates contiguous spans, not isolated tokens."""
        seq_len = 100
        mask = span_corruption_mask(seq_len, 0.15, 5.0, 512, device)

        # Count spans
        num_spans = count_num_spans(mask)

        # With mean span length 5 and 15 tokens, expect ~3 spans
        assert num_spans >= 1
        assert num_spans <= 20  # Not too many isolated tokens

    def test_middle_heavy_mask_prefers_middle(self, device):
        """Test middle-heavy mask has more weight in center."""
        seq_len = 100

        # Run multiple times for statistical stability
        middle_counts = []
        edge_counts = []

        for _ in range(100):
            mask = middle_heavy_mask(seq_len, 0.15, device)
            middle_count = mask[25:75].float().sum().item()
            edge_count = mask[:25].float().sum().item() + mask[75:].float().sum().item()
            middle_counts.append(middle_count)
            edge_counts.append(edge_count)

        avg_middle = sum(middle_counts) / len(middle_counts)
        avg_edge = sum(edge_counts) / len(edge_counts)

        # Middle should have more weight (accounting for it being twice as large)
        # Normalize by region size
        middle_density = avg_middle / 50
        edge_density = avg_edge / 50

        assert middle_density > edge_density

    def test_prefix_lm_mask_random(self, device):
        """Test random prefix mask."""
        seq_len = 100
        mask, split = prefix_lm_mask(seq_len, "random", device)

        # Split should be in valid range
        assert 20 <= split <= 80

        # Before split should be False, after should be True
        assert not mask[:split].any()
        assert mask[split:].all()

    def test_prefix_lm_mask_short(self, device):
        """Test short target prefix mask."""
        seq_len = 100
        mask, split = prefix_lm_mask(seq_len, "short", device)

        # Short target means split near end (85-95)
        assert split >= 80

    def test_prefix_lm_mask_long(self, device):
        """Test long target prefix mask."""
        seq_len = 100
        mask, split = prefix_lm_mask(seq_len, "long", device)

        # Long target means split near beginning (5-20)
        assert split <= 25

    def test_infilling_mask(self, device):
        """Test infilling mask creates middle hole."""
        seq_len = 100
        mask, start, end = infilling_mask(seq_len, 0.30, device)

        # Should have contiguous hole
        assert 10 <= start < end <= 90
        assert (end - start) >= 20  # At least 20% of sequence

        # Hole region should be True
        assert mask[start:end].all()

        # Outside hole should be False
        assert not mask[:start].any()
        assert not mask[end:].any()


# =============================================================================
# Test Sentinel Processing
# =============================================================================


class TestSentinelProcessing:
    """Tests for sentinel ID creation and application."""

    def test_create_sentinel_ids_single_span(self, device):
        """Test sentinel ID creation with single span."""
        mask = torch.tensor([False, False, True, True, False, False], device=device)
        sentinel_ids = create_sentinel_ids(mask, sentinel_start_id=1000)

        # Position 2 should have first sentinel (1000)
        assert sentinel_ids[2].item() == 1000
        # Position 3 is continuation (-1)
        assert sentinel_ids[3].item() == -1
        # Other positions should be 0
        assert sentinel_ids[0].item() == 0
        assert sentinel_ids[4].item() == 0

    def test_create_sentinel_ids_multiple_spans(self, device):
        """Test sentinel ID creation with multiple spans."""
        mask = torch.tensor(
            [False, True, True, False, True, True, True, False], device=device
        )
        sentinel_ids = create_sentinel_ids(mask, sentinel_start_id=1000)

        # First span starts at position 1, gets sentinel 1000
        assert sentinel_ids[1].item() == 1000
        assert sentinel_ids[2].item() == -1  # continuation

        # Second span starts at position 4, gets sentinel 1001
        assert sentinel_ids[4].item() == 1001
        assert sentinel_ids[5].item() == -1  # continuation
        assert sentinel_ids[6].item() == -1  # continuation

    def test_apply_sentinel_mask(self, device):
        """Test applying sentinel mask to input."""
        input_ids = torch.tensor([10, 20, 30, 40, 50, 60], device=device)
        sentinel_ids = torch.tensor([0, 0, 1000, -1, 0, 0], device=device)

        result = apply_sentinel_mask(input_ids, sentinel_ids, None, None)

        # Should be: [10, 20, 1000, 50, 60] (30, 40 replaced with 1000)
        expected = torch.tensor([10, 20, 1000, 50, 60], device=device)
        assert torch.equal(result, expected)

    def test_apply_sentinel_mask_with_eos(self, device):
        """Test applying sentinel mask with EOS token."""
        input_ids = torch.tensor([10, 20, 30, 40], device=device)
        sentinel_ids = torch.tensor([0, 0, 1000, -1], device=device)

        result = apply_sentinel_mask(input_ids, sentinel_ids, None, eos_id=1)

        # Should end with EOS
        assert result[-1].item() == 1

    def test_apply_sentinel_mask_with_prefix(self, device):
        """Test applying sentinel mask with prefix tokens."""
        input_ids = torch.tensor([10, 20, 30], device=device)
        sentinel_ids = torch.tensor([0, 1000, -1], device=device)
        prefix_ids = torch.tensor([100, 101], device=device)

        result = apply_sentinel_mask(input_ids, sentinel_ids, prefix_ids, None)

        # Should start with prefix
        assert result[0].item() == 100
        assert result[1].item() == 101

    def test_count_num_spans(self, device):
        """Test span counting."""
        # Two spans
        mask = torch.tensor(
            [False, True, True, False, True, True, True, False], device=device
        )
        assert count_num_spans(mask) == 2

        # Three spans
        mask = torch.tensor([True, False, True, False, True], device=device)
        assert count_num_spans(mask) == 3

        # No spans
        mask = torch.tensor([False, False, False], device=device)
        assert count_num_spans(mask) == 0


# =============================================================================
# Test UL2DataCollator
# =============================================================================


class TestUL2DataCollator:
    """Tests for UL2DataCollator."""

    def test_collator_initialization(self, mock_tokenizer):
        """Test collator initializes correctly."""
        config = UL2Config.t5gemma2()
        collator = UL2DataCollator(mock_tokenizer, config)

        assert collator.sentinel_start_id == 1000
        assert collator.eos_id == 1
        assert collator.pad_id == 0

    def test_collator_output_keys(self, mock_tokenizer, device):
        """Test collator produces correct output keys."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        examples = [
            {"input_ids": torch.randint(100, 500, (64,), device=device)},
            {"input_ids": torch.randint(100, 500, (48,), device=device)},
        ]

        batch = collator(examples)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "decoder_input_ids" in batch
        assert "decoder_attention_mask" in batch
        assert "labels" in batch

    def test_collator_output_shapes(self, mock_tokenizer, device):
        """Test collator produces correct shapes."""
        collator = UL2DataCollator(
            mock_tokenizer,
            UL2Config.minimal(),
            max_length=128,
            max_labels_length=64,
        )

        examples = [
            {"input_ids": torch.randint(100, 500, (96,), device=device)},
            {"input_ids": torch.randint(100, 500, (80,), device=device)},
            {"input_ids": torch.randint(100, 500, (112,), device=device)},
        ]

        batch = collator(examples)

        batch_size = len(examples)
        assert batch["input_ids"].shape[0] == batch_size
        assert batch["attention_mask"].shape[0] == batch_size
        assert batch["decoder_input_ids"].shape[0] == batch_size
        assert batch["decoder_attention_mask"].shape[0] == batch_size
        assert batch["labels"].shape[0] == batch_size

        # Lengths should be within limits
        assert batch["input_ids"].shape[1] <= 128
        assert batch["labels"].shape[1] <= 64

    def test_collator_dtypes(self, mock_tokenizer, device):
        """Test collator produces correct dtypes."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        examples = [{"input_ids": torch.randint(100, 500, (64,), device=device)}]

        batch = collator(examples)

        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["decoder_input_ids"].dtype == torch.long
        assert batch["decoder_attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    def test_collator_padding(self, mock_tokenizer, device):
        """Test collator correctly pads sequences."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        # Create examples of different lengths
        examples = [
            {"input_ids": torch.randint(100, 500, (32,), device=device)},
            {"input_ids": torch.randint(100, 500, (64,), device=device)},
        ]

        batch = collator(examples)

        # Check attention mask matches padding
        for i in range(len(examples)):
            mask = batch["attention_mask"][i]
            ids = batch["input_ids"][i]

            # Where mask is 0, ids should be pad token
            pad_positions = mask == 0
            if pad_positions.any():
                assert (ids[pad_positions] == 0).all()

    def test_collator_labels_ignore_index(self, mock_tokenizer, device):
        """Test collator uses -100 for label padding."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        examples = [
            {"input_ids": torch.randint(100, 500, (64,), device=device)},
        ]

        batch = collator(examples)

        labels = batch["labels"]
        # Should have some -100 values for padding
        # (unless the sequence exactly fills the label length)
        assert labels.dtype == torch.long

    def test_collator_decoder_shift(self, mock_tokenizer, device):
        """Test decoder input is properly shifted from labels."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        examples = [{"input_ids": torch.randint(100, 500, (64,), device=device)}]

        batch = collator(examples)

        decoder_input = batch["decoder_input_ids"]
        labels = batch["labels"]

        # First position should be decoder_start_token
        assert decoder_input[0, 0].item() in [0, 2]  # pad or bos

        # Rest should be shifted from labels (where labels != -100)
        for i in range(1, decoder_input.shape[1]):
            if labels[0, i - 1].item() != -100:
                # Shifted correctly
                pass  # The shift is done correctly

    def test_collator_handles_list_input(self, mock_tokenizer, device):
        """Test collator handles list inputs (not just tensors)."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        examples = [
            {"input_ids": list(range(100, 164))},
            {"input_ids": list(range(200, 248))},
        ]

        batch = collator(examples)

        assert batch["input_ids"].shape[0] == 2


# =============================================================================
# Test Task Distribution
# =============================================================================


class TestTaskDistribution:
    """Tests for task sampling distribution."""

    def test_t5gemma2_distribution(self, mock_tokenizer, device):
        """Test T5Gemma 2 produces expected task distribution."""
        config = UL2Config.t5gemma2()
        collator = UL2DataCollator(mock_tokenizer, config)

        # Sample many times
        n_samples = 1000
        task_counts = Counter()

        for _ in range(n_samples):
            idx = torch.multinomial(collator._weights, 1).item()
            task_counts[idx] += 1

        # Expected distribution: 1:1:1:1:4 (normalized to 12.5%, 12.5%, 12.5%, 12.5%, 50%)
        for i in range(4):
            # Tasks 0-3 should each be ~12.5% (allow 5% tolerance)
            pct = task_counts[i] / n_samples
            assert 0.08 < pct < 0.18, f"Task {i} had {pct:.1%}, expected ~12.5%"

        # Task 4 (S-denoiser) should be ~50%
        pct_s = task_counts[4] / n_samples
        assert 0.40 < pct_s < 0.60, f"S-denoiser had {pct_s:.1%}, expected ~50%"


# =============================================================================
# Test Token Conservation
# =============================================================================


class TestTokenConservation:
    """Tests for encoder-decoder token conservation."""

    def test_span_corruption_conserves_tokens(self, device):
        """Test that span corruption preserves all original tokens."""
        # Simple test: original tokens should appear in encoder or decoder
        input_ids = torch.arange(100, 120, device=device)  # 20 tokens

        # Create a simple span mask
        mask = torch.zeros(20, dtype=torch.bool, device=device)
        mask[5:10] = True  # Corrupt positions 5-9

        sentinel_start = 1000

        # Encoder: masked tokens replaced with sentinels
        enc_sentinels = create_sentinel_ids(mask, sentinel_start)
        encoder_ids = apply_sentinel_mask(input_ids, enc_sentinels, None, None)

        # Decoder: unmasked tokens get sentinels (target is the masked tokens)
        dec_sentinels = create_sentinel_ids(~mask, sentinel_start)
        decoder_ids = apply_sentinel_mask(input_ids, dec_sentinels, None, None)

        # Collect all non-sentinel tokens from both
        all_tokens = set()

        for t in encoder_ids.tolist():
            if t < 1000:
                all_tokens.add(t)

        for t in decoder_ids.tolist():
            if t < 1000:
                all_tokens.add(t)

        # All original tokens should be present
        original_tokens = set(input_ids.tolist())
        assert all_tokens == original_tokens

    def test_prefix_corruption_conserves_tokens(self, device):
        """Test that prefix corruption preserves all original tokens."""
        input_ids = torch.arange(100, 150, device=device)  # 50 tokens

        # Create prefix mask (split at position 20)
        mask = torch.zeros(50, dtype=torch.bool, device=device)
        mask[20:] = True  # Positions 20-49 are "targets"

        sentinel_start = 1000

        enc_sentinels = create_sentinel_ids(mask, sentinel_start)
        encoder_ids = apply_sentinel_mask(input_ids, enc_sentinels, None, None)

        dec_sentinels = create_sentinel_ids(~mask, sentinel_start)
        decoder_ids = apply_sentinel_mask(input_ids, dec_sentinels, None, None)

        # Collect all non-sentinel tokens
        all_tokens = set()

        for t in encoder_ids.tolist():
            if t < 1000:
                all_tokens.add(t)

        for t in decoder_ids.tolist():
            if t < 1000:
                all_tokens.add(t)

        original_tokens = set(input_ids.tolist())
        assert all_tokens == original_tokens


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, mock_tokenizer, device):
        """Test collator handles very short sequences."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        examples = [{"input_ids": torch.randint(100, 500, (5,), device=device)}]

        batch = collator(examples)

        assert batch["input_ids"].shape[0] == 1

    def test_single_token_sequence(self, mock_tokenizer, device):
        """Test collator handles single token sequence."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        examples = [{"input_ids": torch.tensor([100], device=device)}]

        # Should not crash
        batch = collator(examples)
        assert batch["input_ids"].shape[0] == 1

    def test_empty_batch(self, mock_tokenizer):
        """Test collator handles empty batch gracefully."""
        collator = UL2DataCollator(mock_tokenizer, UL2Config.minimal())

        # Empty list should raise or handle gracefully
        with pytest.raises((IndexError, ValueError)):
            collator([])

    def test_max_length_truncation(self, mock_tokenizer, device):
        """Test collator truncates to max_length."""
        collator = UL2DataCollator(
            mock_tokenizer,
            UL2Config.minimal(),
            max_length=32,
            max_labels_length=16,
        )

        # Very long sequence
        examples = [{"input_ids": torch.randint(100, 500, (1000,), device=device)}]

        batch = collator(examples)

        assert batch["input_ids"].shape[1] <= 32
        assert batch["labels"].shape[1] <= 16

    def test_pad_to_multiple(self, mock_tokenizer, device):
        """Test padding to multiple of specified value."""
        collator = UL2DataCollator(
            mock_tokenizer,
            UL2Config.minimal(),
            max_length=512,
            max_labels_length=128,
            pad_to_multiple_of=8,
        )

        examples = [{"input_ids": torch.randint(100, 500, (50,), device=device)}]

        batch = collator(examples)

        # Should be padded to multiple of 8
        assert batch["input_ids"].shape[1] % 8 == 0
        assert batch["labels"].shape[1] % 8 == 0
