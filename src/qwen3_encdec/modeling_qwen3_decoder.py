"""Qwen3 Decoder with merged self/cross attention.

This module implements a decoder based on the Qwen3 architecture with merged
attention following the T5Gemma 2 pattern. Merged attention combines self-attention
and cross-attention into a single operation:

- Q: computed from decoder hidden states only
- K, V: computed from concatenation of [decoder_hidden, encoder_hidden]

This allows reusing pretrained Qwen3 attention weights directly.

Architecture:
- RMSNorm for layer normalization
- RoPE (Rotary Position Embeddings) with separate positions for encoder/decoder
- GQA (Grouped Query Attention) with QK-Norm
- Gated MLP with SiLU activation
- SDPA support for Flash Attention
- KV cache for efficient generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_encoder import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.get_logger(__name__)


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
class Qwen3DecoderOutput(ModelOutput):
    """
    Output type for Qwen3Decoder.

    Args:
        last_hidden_state: Final layer hidden states [batch, dec_len, hidden_size].
        past_key_values: Cached KV states for incremental decoding.
            Each element is ((dec_key, dec_value), (enc_key, enc_value)).
        hidden_states: All hidden states if output_hidden_states=True.
        attentions: All attention weights if output_attentions=True.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# =============================================================================
# Merged Attention
# =============================================================================


class Qwen3MergedAttention(nn.Module):
    """
    Merged self/cross attention for Qwen3 Decoder.

    This combines self-attention and cross-attention into a single operation:
    - Q comes from decoder hidden states only
    - K, V come from concatenation of [decoder hidden, encoder output]

    This allows reusing pretrained attention weights from Qwen3.

    Args:
        config: Model configuration.
        layer_idx: Layer index (for debugging/logging).
    """

    def __init__(self, config: Qwen3EncoderDecoderConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups

        # Projection layers (same weights for self and cross attention)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # QK-Norm layers
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = Qwen3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        Forward pass for merged attention.

        Args:
            decoder_hidden_states: [batch, dec_len, hidden_size]
            encoder_hidden_states: [batch, enc_len, hidden_size]
            decoder_attention_mask: Padding mask for decoder [batch, dec_len]
            encoder_attention_mask: Padding mask for encoder [batch, enc_len]
            decoder_position_ids: Position IDs for decoder [batch, dec_len]
            encoder_position_ids: Position IDs for encoder [batch, enc_len]
            past_key_value: Cached KV states ((dec_k, dec_v), (enc_k, enc_v))
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights, past_key_value).
        """
        batch_size, dec_len, _ = decoder_hidden_states.shape
        enc_len = encoder_hidden_states.shape[1]
        device = decoder_hidden_states.device
        dtype = decoder_hidden_states.dtype

        # Create position IDs if not provided
        if decoder_position_ids is None:
            # For incremental decoding, past_key_value tells us the offset
            if past_key_value is not None:
                past_dec_len = past_key_value[0][0].shape[2]
                decoder_position_ids = torch.arange(
                    past_dec_len, past_dec_len + dec_len, device=device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                decoder_position_ids = torch.arange(dec_len, device=device)
                decoder_position_ids = decoder_position_ids.unsqueeze(0).expand(batch_size, -1)

        if encoder_position_ids is None:
            encoder_position_ids = torch.arange(enc_len, device=device)
            encoder_position_ids = encoder_position_ids.unsqueeze(0).expand(batch_size, -1)

        # === Compute Q from decoder ===
        query_states = self.q_proj(decoder_hidden_states)
        query_states = query_states.view(
            batch_size, dec_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # === Compute K, V ===
        # For decoder portion
        dec_key_states = self.k_proj(decoder_hidden_states)
        dec_value_states = self.v_proj(decoder_hidden_states)
        dec_key_states = dec_key_states.view(
            batch_size, dec_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        dec_value_states = dec_value_states.view(
            batch_size, dec_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # For encoder portion (only compute if not cached)
        encoder_kv_cached = past_key_value is not None and past_key_value[1] is not None
        if encoder_kv_cached:
            # Encoder KV is cached (already has QK-Norm applied)
            enc_key_states = past_key_value[1][0]
            enc_value_states = past_key_value[1][1]
        else:
            # Compute encoder KV
            enc_key_states = self.k_proj(encoder_hidden_states)
            enc_value_states = self.v_proj(encoder_hidden_states)
            enc_key_states = enc_key_states.view(
                batch_size, enc_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            enc_value_states = enc_value_states.view(
                batch_size, enc_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

        # === Apply QK-Norm ===
        query_states = self.q_norm(query_states)
        dec_key_states = self.k_norm(dec_key_states)
        # Only apply QK-Norm to encoder keys if not cached (cached keys already normalized)
        if not encoder_kv_cached:
            enc_key_states = self.k_norm(enc_key_states)

        # === Apply RoPE ===
        # Query gets decoder positions
        dec_cos, dec_sin = self.rotary_emb(query_states, decoder_position_ids)
        query_states, dec_key_states = apply_rotary_pos_emb(
            query_states, dec_key_states, dec_cos, dec_sin
        )

        # Encoder keys get encoder positions (only if not cached)
        if not encoder_kv_cached:
            enc_cos, enc_sin = self.rotary_emb(enc_key_states, encoder_position_ids)
            # Only apply to keys (no query for encoder)
            enc_key_states = (enc_key_states * enc_cos) + (self._rotate_half(enc_key_states) * enc_sin)

        # === Handle KV Cache ===
        if past_key_value is not None:
            # Append new decoder KV to cached decoder KV
            past_dec_key, past_dec_value = past_key_value[0]
            dec_key_states = torch.cat([past_dec_key, dec_key_states], dim=2)
            dec_value_states = torch.cat([past_dec_value, dec_value_states], dim=2)

        if use_cache:
            # Cache structure: ((dec_k, dec_v), (enc_k, enc_v))
            past_key_value = (
                (dec_key_states, dec_value_states),
                (enc_key_states, enc_value_states),
            )

        # Get actual lengths after caching
        cached_dec_len = dec_key_states.shape[2]

        # === Concatenate K, V for merged attention ===
        key_states = torch.cat([dec_key_states, enc_key_states], dim=2)
        value_states = torch.cat([dec_value_states, enc_value_states], dim=2)

        # === Repeat KV for GQA ===
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # === Create merged attention mask ===
        attn_mask = self._create_merged_attention_mask(
            query_len=dec_len,
            cached_dec_len=cached_dec_len,
            enc_len=enc_len,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
        )

        # === Compute attention ===
        if output_attentions:
            attn_output, attn_weights = self._eager_attention(
                query_states, key_states, value_states, attn_mask
            )
        else:
            attn_output = self._sdpa_attention(
                query_states, key_states, value_states, attn_mask
            )
            attn_weights = None

        # === Output projection ===
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, dec_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value if use_cache else None

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _create_merged_attention_mask(
        self,
        query_len: int,
        cached_dec_len: int,
        enc_len: int,
        decoder_attention_mask: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Create merged attention mask for decoder.

        The mask has shape [batch, 1, query_len, cached_dec_len + enc_len] where:
        - Decoder-to-decoder: causal (lower triangular)
        - Decoder-to-encoder: full (all ones, respecting encoder padding)

        For incremental decoding (query_len=1), only the last row matters.
        """
        total_kv_len = cached_dec_len + enc_len

        # Start with zeros (attend everywhere)
        mask = torch.zeros(
            batch_size, 1, query_len, total_kv_len,
            dtype=dtype, device=device
        )

        # === Causal mask for decoder-to-decoder ===
        if query_len > 1:
            # Full sequence: create causal mask
            # Position i can attend to positions 0..i
            causal_mask = torch.triu(
                torch.ones(query_len, cached_dec_len, device=device, dtype=dtype),
                diagonal=1
            )
            mask[:, :, :, :cached_dec_len] = causal_mask * torch.finfo(dtype).min
        else:
            # Incremental decoding: last token can attend to all cached decoder tokens
            # No masking needed for decoder portion
            pass

        # === Apply encoder padding mask ===
        if encoder_attention_mask is not None:
            # encoder_attention_mask: [batch, enc_len], 1=attend, 0=mask
            enc_mask = (1.0 - encoder_attention_mask.to(dtype)) * torch.finfo(dtype).min
            enc_mask = enc_mask[:, None, None, :]  # [batch, 1, 1, enc_len]
            mask[:, :, :, cached_dec_len:] = mask[:, :, :, cached_dec_len:] + enc_mask

        # === Apply decoder padding mask ===
        if decoder_attention_mask is not None and query_len > 1:
            # For padded decoder inputs, mask out attention to padding tokens
            # decoder_attention_mask: [batch, dec_len], 1=attend, 0=mask
            # We need to mask K positions, not Q positions
            # This gets complex with caching; for now, assume no decoder padding during generation
            dec_mask = (1.0 - decoder_attention_mask.to(dtype)) * torch.finfo(dtype).min
            dec_mask = dec_mask[:, None, None, :]  # [batch, 1, 1, dec_len]
            # Only apply to the current decoder portion
            if cached_dec_len == query_len:
                mask[:, :, :, :cached_dec_len] = mask[:, :, :, :cached_dec_len] + dec_mask

        return mask

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention using SDPA (supports Flash Attention)."""
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,  # We handle causality via the mask
        )
        return attn_output

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention manually (for returning attention weights)."""
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply attention mask
        attn_weights = attn_weights + attn_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if self.training and self.config.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


# =============================================================================
# Decoder Layer
# =============================================================================


class Qwen3DecoderLayer(nn.Module):
    """
    Single decoder layer with merged attention.

    Architecture:
        x, enc -> RMSNorm(x) -> MergedAttn(x, enc) -> + -> RMSNorm -> MLP -> +
        |______________________________________________|  |___________________|

    Args:
        config: Model configuration.
        layer_idx: Layer index.
    """

    def __init__(self, config: Qwen3EncoderDecoderConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-norm before attention
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Merged self/cross attention
        self.self_attn = Qwen3MergedAttention(config, layer_idx)

        # Pre-norm before MLP
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MLP
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        Forward pass for decoder layer.

        Args:
            decoder_hidden_states: Decoder input [batch, dec_len, hidden_size].
            encoder_hidden_states: Encoder output [batch, enc_len, hidden_size].
            decoder_attention_mask: Padding mask for decoder.
            encoder_attention_mask: Padding mask for encoder.
            decoder_position_ids: Position IDs for decoder.
            encoder_position_ids: Position IDs for encoder.
            past_key_value: Cached KV for incremental decoding.
            use_cache: Whether to return cache.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights, past_key_value).
        """
        residual = decoder_hidden_states

        # Pre-norm + Merged Attention
        decoder_hidden_states = self.input_layernorm(decoder_hidden_states)
        attn_output, attn_weights, present_key_value = self.self_attn(
            decoder_hidden_states,
            encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_position_ids=encoder_position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        decoder_hidden_states = residual + attn_output

        # Pre-norm + MLP
        residual = decoder_hidden_states
        decoder_hidden_states = self.post_attention_layernorm(decoder_hidden_states)
        decoder_hidden_states = self.mlp(decoder_hidden_states)
        decoder_hidden_states = residual + decoder_hidden_states

        return decoder_hidden_states, attn_weights, present_key_value


# =============================================================================
# Full Decoder
# =============================================================================


class Qwen3DecoderPreTrainedModel(PreTrainedModel):
    """
    Base class for Qwen3 Decoder models.

    Provides weight initialization and config handling.
    """

    config_class = Qwen3EncoderDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following Qwen3 conventions."""
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Decoder(Qwen3DecoderPreTrainedModel):
    """
    Qwen3-based decoder with merged self/cross attention.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: Qwen3EncoderDecoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)

        # Token embeddings (will be tied with encoder)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set input embeddings."""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, Qwen3DecoderOutput]:
        """
        Forward pass for decoder.

        Args:
            input_ids: Decoder input token IDs [batch, dec_len].
            encoder_hidden_states: Encoder outputs [batch, enc_len, hidden_size].
            attention_mask: Decoder padding mask [batch, dec_len].
            encoder_attention_mask: Encoder padding mask [batch, enc_len].
            position_ids: Decoder position IDs [batch, dec_len].
            encoder_position_ids: Encoder position IDs [batch, enc_len].
            inputs_embeds: Pre-computed embeddings (alternative to input_ids).
            past_key_values: Cached KV states for incremental decoding.
            use_cache: Whether to return cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dataclass.

        Returns:
            Decoder outputs with last_hidden_state and optional fields.
        """
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for decoder")

        # Get embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must specify either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, dec_len, _ = inputs_embeds.shape
        enc_len = encoder_hidden_states.shape[1]
        device = inputs_embeds.device

        # Prepare position IDs
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                # Incremental decoding: offset by cached length
                past_length = past_key_values[0][0][0].shape[2]
                position_ids = torch.arange(
                    past_length, past_length + dec_len, device=device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(dec_len, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if encoder_position_ids is None:
            encoder_position_ids = torch.arange(enc_len, device=device)
            encoder_position_ids = encoder_position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds

        # Collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Forward through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, present_key_value = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    encoder_attention_mask,
                    position_ids,
                    encoder_position_ids,
                    past_key_value,
                    use_cache,
                    output_attentions,
                )
            else:
                hidden_states, attn_weights, present_key_value = layer(
                    hidden_states,
                    encoder_hidden_states,
                    decoder_attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    decoder_position_ids=position_ids,
                    encoder_position_ids=encoder_position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if use_cache:
                next_decoder_cache = next_decoder_cache + (present_key_value,)

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_attentions]
                if v is not None
            )

        return Qwen3DecoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
