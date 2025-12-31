"""Qwen3 Encoder implementation with bidirectional attention.

This module implements a bidirectional encoder based on the Qwen3 architecture,
modified for encoder-decoder models (no causal mask).

Key differences from decoder-only Qwen3:
- Bidirectional attention (no causal mask)
- Returns encoder hidden states for cross-attention
- Uses same weight names as Qwen3 for checkpoint loading

Architecture:
- RMSNorm for layer normalization
- RoPE (Rotary Position Embeddings)
- GQA (Grouped Query Attention) with QK-Norm
- Gated MLP with SiLU activation
- SDPA support for Flash Attention
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig

if TYPE_CHECKING:
    from .modeling_qwen3_encdec import Qwen3ForSeq2SeqLM

logger = logging.get_logger(__name__)


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
class Qwen3EncoderOutput(ModelOutput):
    """
    Output type for Qwen3Encoder.

    Args:
        last_hidden_state: Hidden states from the last encoder layer.
            Shape: (batch_size, sequence_length, hidden_size)
        hidden_states: Optional tuple of hidden states from all layers.
            Each has shape: (batch_size, sequence_length, hidden_size)
        attentions: Optional tuple of attention weights from all layers.
            Each has shape: (batch_size, num_heads, seq_len, seq_len)
    """

    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# =============================================================================
# Building Blocks
# =============================================================================


class Qwen3RMSNorm(nn.Module):
    """
    RMS Normalization layer matching Qwen3 implementation.

    RMSNorm normalizes by the root mean square of activations,
    without centering (no mean subtraction).

    Args:
        hidden_size: Dimension of the hidden states.
        eps: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of same shape.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for Qwen3.

    Computes and caches sin/cos embeddings for rotary position encoding.
    Supports dynamic sequence length extension.

    Args:
        dim: Dimension of each attention head (head_dim).
        max_position_embeddings: Maximum sequence length to cache.
        base: Base for the geometric progression of frequencies.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin embeddings
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_seq_len = 0

    def _update_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Update the cached cos/sin embeddings if needed."""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len

            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=torch.float32)

            # Compute frequencies: [seq_len, dim/2]
            freqs = torch.outer(t, self.inv_freq.to(device))

            # Create [seq_len, dim] by concatenating
            emb = torch.cat((freqs, freqs), dim=-1)

            # Cache cos and sin
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for the given positions.

        Args:
            x: Input tensor, used only to determine seq_len, device, and dtype.
                Shape: (batch_size, num_heads, seq_len, head_dim)
            position_ids: Optional position indices. Shape: (batch_size, seq_len)
                If None, uses 0, 1, 2, ..., seq_len-1.

        Returns:
            Tuple of (cos, sin) tensors for rotary embedding application.
            Each has shape: (batch_size, 1, seq_len, head_dim) or (1, 1, seq_len, head_dim)
        """
        seq_len = x.shape[2]

        # Update cache if needed - must cover max position_id, not just seq_len
        if position_ids is not None:
            max_pos = int(position_ids.max()) + 1  # +1 because positions are 0-indexed
            cache_len = max(seq_len, max_pos)
        else:
            cache_len = seq_len
        self._update_cos_sin_cache(cache_len, x.device, x.dtype)

        if position_ids is not None:
            # Index into cached values using position_ids
            cos = self._cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, D]
            sin = self._sin_cached[position_ids].unsqueeze(1)  # [B, 1, S, D]
        else:
            # Use sequential positions
            cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
            sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    Splits the last dimension in half and swaps with sign change:
    [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]

    Args:
        x: Input tensor of shape (..., head_dim).

    Returns:
        Rotated tensor of same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    RoPE formula: x_rotated = x * cos + rotate_half(x) * sin

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        cos: Cosine embeddings of shape (batch, 1, seq_len, head_dim) or broadcastable.
        sin: Sine embeddings of shape (batch, 1, seq_len, head_dim) or broadcastable.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query head count for GQA.

    For GQA, we have fewer KV heads than query heads. This function
    repeats each KV head n_rep times to match query dimensions.

    Args:
        hidden_states: KV tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        n_rep: Number of times to repeat each head.

    Returns:
        Expanded tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


# =============================================================================
# Attention Layer
# =============================================================================


class Qwen3EncoderAttention(nn.Module):
    """
    Bidirectional multi-head attention for Qwen3 Encoder.

    Features:
    - Grouped Query Attention (GQA)
    - QK-Norm: RMSNorm applied to Q and K after projection, before RoPE
    - RoPE (Rotary Position Embeddings)
    - SDPA support for Flash Attention
    - No causal mask (bidirectional attention)

    Args:
        config: Model configuration.
        layer_idx: Index of this layer (for logging/debugging).
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for bidirectional attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional mask of shape (batch, seq_len) where
                1 = attend, 0 = mask out. Will be converted to additive mask.
            position_ids: Optional position indices of shape (batch, seq_len).
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of:
            - output: Attention output of shape (batch, seq_len, hidden_size).
            - attn_weights: Optional attention weights if output_attentions=True.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply QK-Norm (per-head normalization)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Repeat KV heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        if output_attentions:
            # Manual attention to get weights
            attn_output, attn_weights = self._eager_attention(
                query_states, key_states, value_states, attention_mask
            )
        else:
            # Use SDPA for efficiency
            attn_output = self._sdpa_attention(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None

        # Reshape back to [batch, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute attention using SDPA (supports Flash Attention).

        Args:
            query: Shape (batch, num_heads, seq_len, head_dim).
            key: Shape (batch, num_heads, seq_len, head_dim).
            value: Shape (batch, num_heads, seq_len, head_dim).
            attention_mask: Optional mask of shape (batch, seq_len).

        Returns:
            Attention output of shape (batch, num_heads, seq_len, head_dim).
        """
        # Convert attention mask to SDPA format
        # SDPA expects: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        # where masked positions have -inf
        attn_mask = None
        if attention_mask is not None:
            # Input mask: (batch, seq_len) where 1=attend, 0=mask
            # SDPA mask: (batch, 1, 1, seq_len) where -inf=mask, 0=attend
            attn_mask = attention_mask[:, None, None, :].to(query.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(query.dtype).min

        # Use SDPA - is_causal=False for bidirectional attention
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,  # Key difference from decoder!
        )

        return attn_output

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention manually (for returning attention weights).

        Args:
            query: Shape (batch, num_heads, seq_len, head_dim).
            key: Shape (batch, num_heads, seq_len, head_dim).
            value: Shape (batch, num_heads, seq_len, head_dim).
            attention_mask: Optional mask of shape (batch, seq_len).

        Returns:
            Tuple of (output, attention_weights).
        """
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply attention mask
        if attention_mask is not None:
            # Expand mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = attention_mask[:, None, None, :].to(attn_weights.dtype)
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        if self.training and self.config.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


# =============================================================================
# MLP Layer
# =============================================================================


class Qwen3MLP(nn.Module):
    """
    Gated MLP layer for Qwen3.

    Uses SwiGLU activation: output = down_proj(act(gate_proj(x)) * up_proj(x))

    Args:
        config: Model configuration.
    """

    def __init__(self, config: Qwen3EncoderDecoderConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated MLP.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of same shape.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# =============================================================================
# Encoder Layer
# =============================================================================


class Qwen3EncoderLayer(nn.Module):
    """
    Single encoder layer for Qwen3.

    Architecture:
    - Pre-norm + Bidirectional Attention + Residual
    - Pre-norm + MLP + Residual

    Args:
        config: Model configuration.
        layer_idx: Index of this layer.
    """

    def __init__(self, config: Qwen3EncoderDecoderConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.self_attn = Qwen3EncoderAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for encoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (output, optional attention_weights).
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


# =============================================================================
# Full Encoder
# =============================================================================


class Qwen3EncoderPreTrainedModel(PreTrainedModel):
    """
    Base class for Qwen3 Encoder models.

    Provides weight initialization and config handling.
    """

    config_class = Qwen3EncoderDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3EncoderLayer"]

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


class Qwen3Encoder(Qwen3EncoderPreTrainedModel):
    """
    Qwen3 Encoder with bidirectional attention.

    This is the core encoder module that takes input_ids or inputs_embeds
    and returns encoded hidden states.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: Qwen3EncoderDecoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Qwen3EncoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input embeddings."""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, Qwen3EncoderOutput]:
        """
        Forward pass for the encoder.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).
                1 for tokens to attend to, 0 for masked tokens.
            position_ids: Position indices of shape (batch, seq_len).
            inputs_embeds: Optional pre-computed embeddings of shape
                (batch, seq_len, hidden_size). Mutually exclusive with input_ids.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a ModelOutput instead of tuple.

        Returns:
            Qwen3EncoderOutput or tuple containing:
            - last_hidden_state: Final layer hidden states
            - hidden_states: Optional tuple of all layer hidden states
            - attentions: Optional tuple of all layer attention weights
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        # Create position_ids if not provided
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds

        # Collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Process through encoder layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    output_attentions,
                )
            else:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None
            )

        return Qwen3EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Qwen3EncoderModel(Qwen3EncoderPreTrainedModel):
    """
    Wrapper class for HuggingFace compatibility.

    Provides a consistent interface with other HuggingFace encoder models.
    The main encoder logic is in `self.model`.
    """

    def __init__(self, config: Qwen3EncoderDecoderConfig) -> None:
        super().__init__(config)
        self.model = Qwen3Encoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embeddings."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input embeddings."""
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, Qwen3EncoderOutput]:
        """Forward pass - delegates to self.model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @classmethod
    def from_seq2seq(cls, seq2seq_model: "Qwen3ForSeq2SeqLM") -> "Qwen3EncoderModel":
        """
        Extract encoder from a trained Seq2Seq model.

        This is useful for extracting the trained encoder after UL2 training
        to use as a standalone embedding model.

        Args:
            seq2seq_model: Trained Qwen3ForSeq2SeqLM model.

        Returns:
            Encoder-only model with weights copied from the seq2seq model.

        Example:
            ```python
            # After training
            seq2seq = Qwen3ForSeq2SeqLM.from_pretrained("path/to/trained")
            encoder = Qwen3EncoderModel.from_seq2seq(seq2seq)
            encoder.save_pretrained("path/to/encoder")
            ```
        """
        # Import here to avoid circular imports
        from .modeling_qwen3_encdec import Qwen3ForSeq2SeqLM as Seq2SeqLM

        if not isinstance(seq2seq_model, Seq2SeqLM):
            raise TypeError(
                f"Expected Qwen3ForSeq2SeqLM, got {type(seq2seq_model).__name__}"
            )

        encoder_model = cls(seq2seq_model.config)
        encoder_model.model.load_state_dict(seq2seq_model.model.encoder.state_dict())
        return encoder_model
