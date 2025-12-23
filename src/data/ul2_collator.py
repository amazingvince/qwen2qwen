"""
UL2 Data Collator for Qwen3 Encoder-Decoder.

Adapted from UL2_5 (https://github.com/pszemraj/UL2_5 - Apache-2.0 License).

Provides batch collation for UL2-style denoising objectives, with output
format compatible with Qwen3ForSeq2SeqLM.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from .ul2_torch import (
    DenoiserSpec,
    Task,
    UL2Config,
    apply_sentinel_mask,
    create_sentinel_ids,
    infilling_mask,
    middle_heavy_mask,
    prefix_lm_mask,
    span_corruption_mask,
)


def _get_sentinel_start_id(tokenizer: Any) -> int:
    """Get the first sentinel token ID (<extra_id_0>)."""
    # Use our tokenizer's method if available
    if hasattr(tokenizer, "get_sentinel_token_id"):
        return tokenizer.get_sentinel_token_id(0)

    # Fallback: try to get from original_vocab_size
    if hasattr(tokenizer, "original_vocab_size"):
        return tokenizer.original_vocab_size

    # Last fallback: search for extra_id tokens
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
            if token_id != tokenizer.unk_token_id:
                return token_id
        except Exception:
            pass

    raise ValueError(
        "Could not determine sentinel token start ID. "
        "Tokenizer must have get_sentinel_token_id() method or "
        "original_vocab_size attribute."
    )


class _TokenizerShim:
    """Shim tokenizer to ensure UL2_5 can resolve sentinel IDs consistently."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._sentinel_start_id = _get_sentinel_start_id(tokenizer)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<extra_id_0>":
            return self._sentinel_start_id
        return self._tokenizer.convert_tokens_to_ids(token)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def _ul25_supported_fields(UL25Config: Any) -> List[str]:
    if hasattr(UL25Config, "model_fields"):
        return list(UL25Config.model_fields.keys())
    if hasattr(UL25Config, "__fields__"):
        return list(UL25Config.__fields__.keys())
    try:
        params = inspect.signature(UL25Config).parameters
        return [name for name in params.keys() if name != "self"]
    except (TypeError, ValueError):
        return []


class _LocalUL2DataCollator:
    """
    PyTorch-native UL2 Data Collator for Qwen3 Encoder-Decoder.

    Follows HuggingFace DataCollator conventions for use with
    Trainer and DataLoader.

    Output format is compatible with Qwen3ForSeq2SeqLM:
    - input_ids: Encoder input (corrupted sequence with sentinels)
    - attention_mask: Encoder attention mask
    - decoder_input_ids: Decoder input (shifted labels)
    - decoder_attention_mask: Decoder attention mask
    - labels: Target tokens with -100 for padding

    Args:
        tokenizer: Qwen3EncoderDecoderTokenizer with sentinel tokens.
        config: UL2Config specifying denoiser mixture.
        max_length: Maximum encoder input length.
        max_labels_length: Maximum decoder target length.
        pad_to_multiple_of: Pad sequences to multiple of this value.
        decoder_start_token_id: Token to prepend to decoder input.
            If None, uses tokenizer.bos_token_id or pad_token_id.

    Example:
        >>> from src.data import UL2DataCollator, UL2Config
        >>> from src.qwen3_encdec import Qwen3EncoderDecoderTokenizer
        >>>
        >>> tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>> collator = UL2DataCollator(tokenizer, UL2Config.t5gemma2())
        >>>
        >>> # With DataLoader
        >>> dataloader = DataLoader(dataset, collate_fn=collator, batch_size=32)
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[UL2Config] = None,
        max_length: int = 512,
        max_labels_length: int = 128,
        pad_to_multiple_of: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or UL2Config.t5gemma2()
        self.max_length = max_length
        self.max_labels_length = max_labels_length
        self.pad_to_multiple_of = pad_to_multiple_of

        # Token IDs
        self.sentinel_start_id = _get_sentinel_start_id(tokenizer)
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id or 0

        # Decoder start token
        if decoder_start_token_id is not None:
            self.decoder_start_token_id = decoder_start_token_id
        elif hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            self.decoder_start_token_id = tokenizer.bos_token_id
        else:
            self.decoder_start_token_id = self.pad_id

        # Pre-encode prefixes (store as CPU tensors)
        self._prefix_cache: Dict[str, Tensor] = {}
        for spec in self.config.denoisers:
            if spec.prefix and spec.prefix not in self._prefix_cache:
                ids = tokenizer.encode(spec.prefix, add_special_tokens=False)
                self._prefix_cache[spec.prefix] = torch.tensor(ids, dtype=torch.long)

        # Sampling weights as tensor
        self._weights = torch.tensor(self.config.weights, dtype=torch.float32)

    def _get_prefix_ids(self, prefix: str, device: torch.device) -> Optional[Tensor]:
        """Get prefix token IDs on correct device."""
        if not prefix:
            return None
        base_tensor = self._prefix_cache.get(prefix)
        if base_tensor is not None:
            return base_tensor.to(device)
        return None

    def _sample_r(self, spec: DenoiserSpec) -> float:
        """Sample corruption rate."""
        if spec.variable_r:
            lo, hi = spec.r_bounds
            return lo + (hi - lo) * torch.rand(1).item()
        return spec.r

    def _generate_mask(
        self,
        seq_len: int,
        spec: DenoiserSpec,
        device: torch.device,
    ) -> Tensor:
        """Generate corruption mask based on task type."""
        r = self._sample_r(spec)

        if spec.task == Task.SPAN:
            return span_corruption_mask(seq_len, r, spec.mu, spec.max_spans, device)
        elif spec.task == Task.SPAN_MIDDLE:
            return middle_heavy_mask(seq_len, r, device)
        elif spec.task == Task.PREFIX_RANDOM:
            return prefix_lm_mask(seq_len, "random", device)[0]
        elif spec.task == Task.PREFIX_SHORT:
            return prefix_lm_mask(seq_len, "short", device)[0]
        elif spec.task == Task.PREFIX_LONG:
            return prefix_lm_mask(seq_len, "long", device)[0]
        elif spec.task == Task.INFILLING:
            return infilling_mask(seq_len, r, device)[0]
        else:
            return span_corruption_mask(seq_len, r, spec.mu, spec.max_spans, device)

    def _process_single(
        self,
        input_ids: Tensor,
        denoiser_idx: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Process single sequence with UL2 denoising."""
        device = input_ids.device
        seq_len = input_ids.shape[0]

        # Sample denoiser
        if denoiser_idx is None:
            denoiser_idx = torch.multinomial(self._weights, 1).item()

        spec = self.config.denoisers[denoiser_idx]
        prefix_ids = self._get_prefix_ids(spec.prefix, device)

        # Generate mask
        mask = self._generate_mask(seq_len, spec, device)

        # Create sentinel IDs
        # For encoder: masked spans become sentinels
        # For decoder: unmasked spans become sentinels (the targets)
        enc_sentinels = create_sentinel_ids(mask, self.sentinel_start_id)
        dec_sentinels = create_sentinel_ids(~mask, self.sentinel_start_id)

        # Apply masks
        encoder_ids = apply_sentinel_mask(
            input_ids, enc_sentinels, prefix_ids, self.eos_id
        )
        decoder_ids = apply_sentinel_mask(input_ids, dec_sentinels, None, self.eos_id)

        return {
            "encoder_ids": encoder_ids,
            "decoder_ids": decoder_ids,
        }

    def _pad_to_multiple(self, length: int) -> int:
        """Round up to multiple if specified."""
        if self.pad_to_multiple_of is None:
            return length
        return (
            (length + self.pad_to_multiple_of - 1)
            // self.pad_to_multiple_of
            * self.pad_to_multiple_of
        )

    def _shift_right(self, labels: Tensor) -> Tensor:
        """
        Shift labels right for decoder input.

        Prepends decoder_start_token_id and shifts all tokens right.
        """
        batch_size, seq_len = labels.shape
        device = labels.device

        shifted = torch.full_like(labels, self.pad_id)
        shifted[:, 0] = self.decoder_start_token_id

        if seq_len > 1:
            shifted[:, 1:] = labels[:, :-1]

        # Replace -100 (ignore index) with pad token
        shifted[shifted == -100] = self.pad_id

        return shifted

    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Tensor]:
        """
        Collate batch of examples.

        Args:
            examples: List of dicts with "input_ids" (Tensor or list).

        Returns:
            Dict with:
            - input_ids: Encoder input [batch, enc_len]
            - attention_mask: Encoder attention mask [batch, enc_len]
            - decoder_input_ids: Decoder input [batch, dec_len]
            - decoder_attention_mask: Decoder attention mask [batch, dec_len]
            - labels: Target tokens [batch, dec_len]
        """
        # Determine device from first example
        first_ids = examples[0]["input_ids"]
        if isinstance(first_ids, Tensor):
            device = first_ids.device
        else:
            device = torch.device("cpu")

        # Batch sample all denoiser indices at once
        batch_size = len(examples)
        denoiser_indices = (
            torch.multinomial(self._weights.expand(batch_size, -1), 1)
            .squeeze(-1)
            .tolist()
        )

        # Process each example
        processed = []
        for i, ex in enumerate(examples):
            input_ids = ex["input_ids"]

            # Convert to tensor if needed
            if not isinstance(input_ids, Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

            # Flatten if 2D
            if input_ids.dim() == 2:
                input_ids = input_ids.squeeze(0)

            input_ids = input_ids.to(device)
            processed.append(self._process_single(input_ids, denoiser_indices[i]))

        # Compute padded lengths
        max_enc = min(
            self.max_length, max(p["encoder_ids"].shape[0] for p in processed)
        )
        max_dec = min(
            self.max_labels_length, max(p["decoder_ids"].shape[0] for p in processed)
        )

        max_enc = self._pad_to_multiple(max_enc)
        max_dec = self._pad_to_multiple(max_dec)

        # Allocate output tensors
        input_ids = torch.full(
            (batch_size, max_enc), self.pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_enc), dtype=torch.long, device=device
        )
        labels = torch.full(
            (batch_size, max_dec), -100, dtype=torch.long, device=device
        )

        # Fill tensors
        for i, p in enumerate(processed):
            enc = p["encoder_ids"]
            dec = p["decoder_ids"]

            enc_len = min(enc.shape[0], max_enc)
            dec_len = min(dec.shape[0], max_dec)

            input_ids[i, :enc_len] = enc[:enc_len]
            attention_mask[i, :enc_len] = 1
            labels[i, :dec_len] = dec[:dec_len]

        # Create decoder inputs by shifting labels right
        decoder_input_ids = self._shift_right(labels)

        # Decoder attention mask (1 where labels != -100, or where decoder_input_ids != pad)
        decoder_attention_mask = (labels != -100).long()
        # Also include the decoder start token position
        decoder_attention_mask[:, 0] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


class UL2DataCollator:
    """
    UL2 Data Collator wrapper.

    Uses UL2_5 when available (pinned dependency) and falls back to the local
    implementation when UL2_5 is not installed or when forced. Set
    `use_ul2_5=False` or `QWEN3_UL2_IMPL=local` to force the local path.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[Union[UL2Config, Any]] = None,
        max_length: int = 512,
        max_labels_length: int = 128,
        pad_to_multiple_of: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_ul2_5: Optional[bool] = None,
        ul2_length_adaptive: Optional[bool] = None,
        ul2_boundary_snapping: Optional[bool] = None,
        ul2_curriculum_start: Optional[List[float]] = None,
        ul2_curriculum_end: Optional[List[float]] = None,
        collate_on_cpu: bool = False,
        return_task_info: bool = False,
    ):
        self.tokenizer = tokenizer
        self.config = config or UL2Config.t5gemma2()
        self.max_length = max_length
        self.max_labels_length = max_labels_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self._collate_on_cpu = collate_on_cpu
        self._ul2_length_adaptive = ul2_length_adaptive
        self._ul2_boundary_snapping = ul2_boundary_snapping
        self._ul2_curriculum_start = ul2_curriculum_start
        self._ul2_curriculum_end = ul2_curriculum_end
        if decoder_start_token_id is not None:
            self.decoder_start_token_id = decoder_start_token_id
        elif hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            self.decoder_start_token_id = tokenizer.bos_token_id
        else:
            self.decoder_start_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self.return_task_info = return_task_info

        env_choice = os.getenv("QWEN3_UL2_IMPL", "").strip().lower()
        if env_choice == "local":
            use_ul2_5 = False
        elif env_choice in {"ul2_5", "ul25", "ul2-5"}:
            use_ul2_5 = True

        self._use_ul2_5 = use_ul2_5 if use_ul2_5 is not None else True
        self._impl = None
        self._ul25_config = None

        if self._use_ul2_5:
            try:
                from UL2_5.collator_torch import UL25DataCollator
                from UL2_5.config import (
                    DenoiserSpec as UL25DenoiserSpec,
                    Task as UL25Task,
                    UL25Config,
                )
            except ImportError:
                if use_ul2_5:
                    raise
                self._use_ul2_5 = False
            else:
                self._ul25_config = self._coerce_ul25_config(
                    self.config, UL25Config, UL25DenoiserSpec, UL25Task
                )
                shim = _TokenizerShim(tokenizer)
                self._impl = UL25DataCollator(
                    shim,
                    config=self._ul25_config,
                    max_length=max_length,
                    max_labels_length=max_labels_length,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_tensors="pt",
                    return_task_info=return_task_info,
                )
                self._weights = torch.tensor(
                    list(self._ul25_config.weights), dtype=torch.float32
                )

        if not self._use_ul2_5:
            if not isinstance(self.config, UL2Config):
                raise TypeError(
                    "Local UL2 collator requires UL2Config. "
                    "Pass use_ul2_5=True to use UL2_5 configs."
                )
            self._impl = _LocalUL2DataCollator(
                tokenizer=tokenizer,
                config=self.config,
                max_length=max_length,
                max_labels_length=max_labels_length,
                pad_to_multiple_of=pad_to_multiple_of,
                decoder_start_token_id=self.decoder_start_token_id,
            )
            self._weights = self._impl._weights

    def _coerce_ul25_config(
        self,
        config: Union[UL2Config, Any],
        UL25Config,
        UL25DenoiserSpec,
        UL25Task,
    ):
        if isinstance(config, UL25Config):
            return config
        if not isinstance(config, UL2Config):
            raise TypeError(
                "UL2_5 collator expects UL2Config or UL25Config. "
                f"Got {type(config).__name__}"
            )

        denoisers = [
            UL25DenoiserSpec(
                task=UL25Task(spec.task),
                mu=spec.mu,
                r=spec.r,
                max_spans=spec.max_spans,
                prefix=spec.prefix,
                variable_r=spec.variable_r,
                r_bounds=spec.r_bounds,
            )
            for spec in config.denoisers
        ]

        cfg_kwargs: Dict[str, Any] = {
            "denoisers": denoisers,
            "weights": list(config.weights),
        }
        if self._ul2_length_adaptive is not None:
            cfg_kwargs["enable_length_adaptive"] = self._ul2_length_adaptive
        if self._ul2_boundary_snapping is not None:
            cfg_kwargs["enable_boundary_snapping"] = self._ul2_boundary_snapping
        if self._ul2_curriculum_start is not None:
            cfg_kwargs["curriculum_start"] = self._ul2_curriculum_start
        if self._ul2_curriculum_end is not None:
            cfg_kwargs["curriculum_end"] = self._ul2_curriculum_end

        supported = _ul25_supported_fields(UL25Config)
        if supported:
            cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in supported}

        try:
            return UL25Config(**cfg_kwargs)
        except TypeError:
            minimal_kwargs = {
                k: v for k, v in cfg_kwargs.items() if k in {"denoisers", "weights"}
            }
            return UL25Config(**minimal_kwargs)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        if self._collate_on_cpu:
            cpu_examples = []
            for ex in examples:
                ex_cpu = {}
                for key, value in ex.items():
                    if isinstance(value, Tensor):
                        ex_cpu[key] = value.cpu()
                    else:
                        ex_cpu[key] = value
                cpu_examples.append(ex_cpu)
            examples = cpu_examples

        batch = self._impl(examples)

        if self._use_ul2_5:
            # Ensure decoder attention mask is present for compatibility
            if "decoder_attention_mask" not in batch:
                decoder_attention_mask = (batch["labels"] != -100).long()
                if decoder_attention_mask.shape[1] > 0:
                    decoder_attention_mask[:, 0] = 1
                batch["decoder_attention_mask"] = decoder_attention_mask

            # Optionally override decoder start token
            if (
                self.decoder_start_token_id is not None
                and batch["decoder_input_ids"].shape[1] > 0
            ):
                batch["decoder_input_ids"][:, 0] = self.decoder_start_token_id

        if self._collate_on_cpu:
            for key, value in batch.items():
                if isinstance(value, Tensor):
                    batch[key] = value.cpu()

        return batch
