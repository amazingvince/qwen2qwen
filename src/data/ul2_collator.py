"""UL2.5 collator adapter for Qwen3 encoder-decoder training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

try:
    from UL2_5.collator_torch import UL25DataCollator as _UL25DataCollator
    from UL2_5.config import DenoiserSpec, Task, UL25Config
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "UL2_5 is required for UL2 training. Install with `pip install -e '.[training]'`."
    ) from exc


def t5gemma2_config(
    *,
    weights: Optional[List[float]] = None,
    enable_length_adaptive: bool = False,
    enable_boundary_snapping: bool = False,
    curriculum_start: Optional[List[float]] = None,
    curriculum_end: Optional[List[float]] = None,
    enable_unpad_encoder: bool = False,
    enable_unpad_decoder: bool = False,
) -> UL25Config:
    """
    T5Gemma 2 UL2 mixture used by this project.

    Mixture: R1, R2, X1, X2, S with default weights 1:1:1:1:4.
    """
    return UL25Config(
        denoisers=[
            DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),  # R1
            DenoiserSpec(task=Task.SPAN, mu=12.0, r=0.50, prefix="[R]"),  # R2
            DenoiserSpec(task=Task.SPAN, mu=32.0, r=0.15, prefix="[X]"),  # X1
            DenoiserSpec(task=Task.SPAN, mu=32.0, r=0.50, prefix="[X]"),  # X2
            DenoiserSpec(task=Task.PREFIX_RANDOM, r=0.75, prefix="[S]"),  # S
        ],
        weights=list(weights) if weights is not None else [1, 1, 1, 1, 4],
        curriculum_start=curriculum_start,
        curriculum_end=curriculum_end,
        enable_length_adaptive=enable_length_adaptive,
        enable_boundary_snapping=enable_boundary_snapping,
        enable_unpad_encoder=enable_unpad_encoder,
        enable_unpad_decoder=enable_unpad_decoder,
    )


def _infer_ul25_sentinel_start_id(tokenizer: Any) -> int:
    """
    UL2_5 expects <extra_id_0> to resolve to the *highest* sentinel token ID.

    Our tokenizer layout uses contiguous sentinel IDs but may not follow the
    T5 ID ordering; normalize by taking the maximum known sentinel ID.
    """
    num_sentinels = getattr(tokenizer, "num_sentinel_tokens", None)
    if (
        isinstance(num_sentinels, int)
        and num_sentinels > 0
        and hasattr(tokenizer, "get_sentinel_token_id")
    ):
        try:
            return int(tokenizer.get_sentinel_token_id(num_sentinels - 1))
        except Exception:
            pass

    unk_id = getattr(tokenizer, "unk_token_id", None)
    last_token = (
        f"<extra_id_{num_sentinels - 1}>" if isinstance(num_sentinels, int) else "<extra_id_99>"
    )

    candidate_ids: List[int] = []
    for token in ("<extra_id_0>", last_token):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if token_id is None:
            continue
        if unk_id is not None and token_id == unk_id:
            continue
        candidate_ids.append(int(token_id))

    if candidate_ids:
        return max(candidate_ids)

    original_vocab_size = getattr(tokenizer, "original_vocab_size", None)
    if (
        isinstance(num_sentinels, int)
        and num_sentinels > 0
        and isinstance(original_vocab_size, int)
        and original_vocab_size >= 0
    ):
        return original_vocab_size + num_sentinels - 1

    raise ValueError(
        "Unable to infer UL2_5 sentinel_start ID. "
        "Tokenizer must support extra_id tokens or expose original_vocab_size/num_sentinel_tokens."
    )


class _TokenizerShim:
    """Shim tokenizer so UL2_5 detects sentinel_start correctly."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._sentinel_start_id = _infer_ul25_sentinel_start_id(tokenizer)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<extra_id_0>":
            return self._sentinel_start_id
        return self._tokenizer.convert_tokens_to_ids(token)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


class UL2DataCollator:
    """Project-facing UL2 collator implemented via UL2_5."""

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[UL25Config] = None,
        max_length: int = 512,
        max_labels_length: int = 128,
        pad_to_multiple_of: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        return_task_info: bool = False,
        collate_on_cpu: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or t5gemma2_config()
        self.max_length = max_length
        self.max_labels_length = max_labels_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self._collate_on_cpu = collate_on_cpu

        if decoder_start_token_id is not None:
            self.decoder_start_token_id = int(decoder_start_token_id)
        elif getattr(tokenizer, "bos_token_id", None) is not None:
            self.decoder_start_token_id = int(tokenizer.bos_token_id)
        else:
            self.decoder_start_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

        self._impl = _UL25DataCollator(
            _TokenizerShim(tokenizer),
            config=self.config,
            max_length=max_length,
            max_labels_length=max_labels_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
            return_task_info=return_task_info,
        )

    @property
    def progress(self) -> float:
        return float(getattr(self._impl, "progress", 0.0))

    @progress.setter
    def progress(self, value: float) -> None:
        if hasattr(self._impl, "progress"):
            self._impl.progress = float(value)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        if self._collate_on_cpu:
            cpu_examples: List[Dict[str, Any]] = []
            for ex in examples:
                ex_cpu: Dict[str, Any] = {}
                for key, value in ex.items():
                    ex_cpu[key] = value.cpu() if isinstance(value, Tensor) else value
                cpu_examples.append(ex_cpu)
            examples = cpu_examples

        batch: Dict[str, Tensor] = self._impl(examples)

        decoder_attention_mask = (batch["labels"] != -100).long()
        if decoder_attention_mask.shape[1] > 0:
            decoder_attention_mask[:, 0] = 1
        batch["decoder_attention_mask"] = decoder_attention_mask

        if batch["decoder_input_ids"].shape[1] > 0:
            batch["decoder_input_ids"][:, 0] = self.decoder_start_token_id

        if self._collate_on_cpu:
            for key, value in list(batch.items()):
                if isinstance(value, Tensor):
                    batch[key] = value.cpu()

        return batch
