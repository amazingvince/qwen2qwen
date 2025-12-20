"""
UL2 Data Collator for Qwen3 Encoder-Decoder.

Adapted from UL2_5 (https://github.com/pszemraj/UL2_5 - Apache-2.0 License).

Provides batch collation for UL2-style denoising objectives, with output
format compatible with Qwen3ForSeq2SeqLM.
"""

from __future__ import annotations

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


class UL2DataCollator:
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
        self.sentinel_start_id = self._get_sentinel_start_id()
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

    def _get_sentinel_start_id(self) -> int:
        """Get the first sentinel token ID (<extra_id_0>)."""
        # Use our tokenizer's method if available
        if hasattr(self.tokenizer, "get_sentinel_token_id"):
            return self.tokenizer.get_sentinel_token_id(0)

        # Fallback: try to get from original_vocab_size
        if hasattr(self.tokenizer, "original_vocab_size"):
            return self.tokenizer.original_vocab_size

        # Last fallback: search for extra_id tokens
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
                if token_id != self.tokenizer.unk_token_id:
                    return token_id
            except Exception:
                pass

        raise ValueError(
            "Could not determine sentinel token start ID. "
            "Tokenizer must have get_sentinel_token_id() method or "
            "original_vocab_size attribute."
        )

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
