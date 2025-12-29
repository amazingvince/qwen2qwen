"""Tests for UL2_5-backed UL2 data pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.data import UL2DataCollator, t5gemma2_config


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
