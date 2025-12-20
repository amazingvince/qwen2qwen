# Story 03: Qwen3 Encoder Implementation (Bidirectional)

## Overview

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-003 |
| **Title** | Qwen3 Encoder Implementation (Bidirectional) |
| **Priority** | P0 - Critical Path |
| **Estimated Effort** | 3-4 days |
| **Dependencies** | Story 01 (Configuration) |
| **Deliverables** | `Qwen3Encoder`, `Qwen3EncoderLayer`, bidirectional attention, unit tests |

---

## Objective

Implement the encoder component of the Qwen3 Encoder-Decoder model. The encoder is essentially Qwen3 with **bidirectional attention** (causal mask removed). All other components (GQA, RoPE, RMSNorm, MLP) remain identical to the original Qwen3 architecture.

---

## Background & Context

### Key Difference from Qwen3
The only fundamental difference between the encoder and Qwen3's decoder layers is:

| Aspect | Qwen3 (Decoder) | Qwen3 Encoder |
|--------|-----------------|---------------|
| **Attention Mask** | Causal (lower triangular) | Bidirectional (no causal mask) |
| **Use Case** | Next-token prediction | Full sequence encoding |

### Architecture to Preserve
From Qwen3-0.6B:
- **28 layers**
- **Grouped Query Attention (GQA)**: 16 query heads, 8 KV heads
- **QK-Norm**: RMSNorm on Q and K before attention
- **RoPE**: Rotary Position Embedding
- **Pre-norm**: RMSNorm before attention and MLP
- **Gated MLP**: SiLU activation with gate projection
- **Sliding Window Attention**: Optional per-layer (configured via `layer_types`)

### Reference Code
Study the HuggingFace Qwen3 implementation:
- `transformers/models/qwen3/modeling_qwen3.py`
- Pay attention to `Qwen3Attention`, `Qwen3MLP`, `Qwen3DecoderLayer`

---

## Technical Requirements

### 1. Encoder Layer Class

#### File: `modeling_qwen3_encoder.py`

```python
"""Qwen3 Encoder with bidirectional attention."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.cache_utils import Cache

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig

logger = logging.get_logger(__name__)


class Qwen3RMSNorm(nn.Module):
    """
    RMSNorm implementation matching Qwen3.
    
    Args:
        hidden_size: Dimensionality of the input.
        eps: Epsilon for numerical stability.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for Qwen3.
    
    Args:
        dim: Dimension of the embedding (typically head_dim).
        max_position_embeddings: Maximum sequence length.
        base: Base for the frequency computation.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, device)
    
    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ):
        """Precompute cos and sin values."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embedding.
        
        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim].
            position_ids: Position indices [batch, seq_len].
            
        Returns:
            Tuple of (cos, sin) tensors for the positions.
        """
        seq_len = position_ids.max() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        
        cos = self.cos_cached[position_ids]  # [batch, seq_len, dim]
        sin = self.sin_cached[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
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
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim].
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim].
        cos: Cosine values [batch, seq_len, head_dim].
        sin: Sine values [batch, seq_len, head_dim].
        
    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads for grouped query attention.
    
    Args:
        hidden_states: KV tensor [batch, num_kv_heads, seq_len, head_dim].
        n_rep: Number of times to repeat each KV head.
        
    Returns:
        Expanded tensor [batch, num_heads, seq_len, head_dim].
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class Qwen3EncoderAttention(nn.Module):
    """
    Bidirectional attention for Qwen3 Encoder.
    
    This is identical to Qwen3Attention except:
    1. No causal mask is applied
    2. Supports attention_mask for padding
    
    Args:
        config: Model configuration.
        layer_idx: Layer index (for logging).
    """
    
    def __init__(self, config: Qwen3EncoderDecoderConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.attention_dropout = config.attention_dropout
        
        # Projection layers
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        
        # QK-Norm (Qwen3 specific)
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
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for bidirectional attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Padding mask [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len].
                           0 for positions to attend, large negative for masked positions.
            position_ids: Position indices [batch, seq_len].
            output_attentions: Whether to return attention weights.
            
        Returns:
            Tuple of (output, attention_weights).
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
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply QK-Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        
        # Repeat KV heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        # Apply attention mask (for padding, NOT causal)
        # NOTE: No causal mask - this is bidirectional attention
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class Qwen3MLP(nn.Module):
    """
    Gated MLP with SiLU activation, matching Qwen3.
    
    Args:
        config: Model configuration.
    """
    
    def __init__(self, config: Qwen3EncoderDecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Uses gated activation: down(act(gate(x)) * up(x))
        """
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Qwen3EncoderLayer(nn.Module):
    """
    Single encoder layer with bidirectional attention.
    
    Architecture:
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |___________________________|  |___________________|
    
    Args:
        config: Model configuration.
        layer_idx: Layer index.
    """
    
    def __init__(self, config: Qwen3EncoderDecoderConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-norm before attention
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Bidirectional self-attention
        self.self_attn = Qwen3EncoderAttention(config, layer_idx)
        
        # Pre-norm before MLP
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        
        # MLP
        self.mlp = Qwen3MLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for encoder layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Padding mask.
            position_ids: Position indices.
            output_attentions: Whether to return attention weights.
            
        Returns:
            Tuple of (output, attention_weights).
        """
        residual = hidden_states
        
        # Pre-norm + Attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output
        
        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights


class Qwen3Encoder(PreTrainedModel):
    """
    Qwen3-based encoder with bidirectional attention.
    
    This encoder uses the same architecture as Qwen3 but with
    bidirectional attention (no causal mask).
    
    Args:
        config: Model configuration.
    """
    
    config_class = Qwen3EncoderDecoderConfig
    
    def __init__(self, config: Qwen3EncoderDecoderConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings (will be tied with decoder and output)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            Qwen3EncoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Return input embeddings."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embeddings."""
        self.embed_tokens = value
    
    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Prepare attention mask for bidirectional attention.
        
        Unlike causal attention, we only need to mask padding tokens.
        
        Args:
            attention_mask: Input mask [batch, seq_len], 1 for valid, 0 for pad.
            batch_size: Batch size.
            seq_len: Sequence length.
            dtype: Tensor dtype.
            device: Tensor device.
            
        Returns:
            Expanded mask [batch, 1, 1, seq_len] with -inf for masked positions.
        """
        if attention_mask is None:
            return None
        
        # Expand mask for attention computation
        # [batch, seq_len] -> [batch, 1, 1, seq_len]
        expanded_mask = attention_mask[:, None, None, :]
        
        # Convert to attention mask format (0 = attend, -inf = mask)
        expanded_mask = (1.0 - expanded_mask.to(dtype)) * torch.finfo(dtype).min
        
        return expanded_mask
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, "Qwen3EncoderOutput"]:
        """
        Forward pass for encoder.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Padding mask [batch, seq_len].
            position_ids: Position indices [batch, seq_len].
            inputs_embeds: Pre-computed embeddings (alternative to input_ids).
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dataclass.
            
        Returns:
            Encoder outputs with last_hidden_state and optional attentions/hidden_states.
        """
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        
        # Prepare position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask (bidirectional, only padding)
        attention_mask = self._prepare_attention_mask(
            attention_mask, batch_size, seq_len, dtype, device
        )
        
        hidden_states = inputs_embeds
        
        # Collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Forward through layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
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
                all_attentions += (attn_weights,)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if return_dict:
            return Qwen3EncoderOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        
        return (hidden_states, all_hidden_states, all_attentions)


# Output dataclass
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput


@dataclass
class Qwen3EncoderOutput(ModelOutput):
    """
    Output type for Qwen3Encoder.
    
    Args:
        last_hidden_state: Final layer hidden states [batch, seq_len, hidden_size].
        hidden_states: All hidden states if output_hidden_states=True.
        attentions: All attention weights if output_attentions=True.
    """
    
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
```

---

## Unit Tests

#### File: `tests/test_encoder.py`

```python
"""Unit tests for Qwen3Encoder."""

import pytest
import torch
import torch.nn as nn

from qwen3_encdec.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from qwen3_encdec.modeling_qwen3_encoder import (
    Qwen3Encoder,
    Qwen3EncoderLayer,
    Qwen3EncoderAttention,
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


class TestQwen3RMSNorm:
    """Test RMSNorm implementation."""
    
    def test_forward_shape(self):
        """Test output shape matches input shape."""
        norm = Qwen3RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_normalization(self):
        """Test that output is normalized."""
        norm = Qwen3RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64) * 100  # Large values
        out = norm(x)
        
        # Check that variance is reduced
        assert out.var() < x.var()


class TestQwen3RotaryEmbedding:
    """Test RoPE implementation."""
    
    def test_forward_shapes(self):
        """Test output shapes."""
        rotary = Qwen3RotaryEmbedding(dim=64, max_position_embeddings=128)
        x = torch.randn(2, 10, 8, 64)  # [batch, seq, heads, dim]
        position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        
        cos, sin = rotary(x, position_ids)
        
        assert cos.shape == (2, 10, 64)
        assert sin.shape == (2, 10, 64)
    
    def test_cache_extension(self):
        """Test that cache extends for longer sequences."""
        rotary = Qwen3RotaryEmbedding(dim=64, max_position_embeddings=64)
        
        # Request longer sequence
        x = torch.randn(1, 100, 8, 64)
        position_ids = torch.arange(100).unsqueeze(0)
        
        cos, sin = rotary(x, position_ids)
        assert cos.shape[1] == 100


class TestApplyRotaryPosEmb:
    """Test RoPE application."""
    
    def test_output_shapes(self):
        """Test that output shapes match input."""
        q = torch.randn(2, 8, 10, 64)  # [batch, heads, seq, dim]
        k = torch.randn(2, 4, 10, 64)
        cos = torch.randn(2, 10, 64)
        sin = torch.randn(2, 10, 64)
        
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestRepeatKV:
    """Test KV head repetition for GQA."""
    
    def test_no_repeat(self):
        """Test identity when n_rep=1."""
        kv = torch.randn(2, 8, 10, 64)
        out = repeat_kv(kv, n_rep=1)
        assert torch.allclose(out, kv)
    
    def test_repeat_2x(self):
        """Test doubling KV heads."""
        kv = torch.randn(2, 4, 10, 64)
        out = repeat_kv(kv, n_rep=2)
        assert out.shape == (2, 8, 10, 64)
        
        # Check that values are repeated
        assert torch.allclose(out[:, 0], out[:, 1])
        assert torch.allclose(out[:, 2], out[:, 3])


class TestQwen3EncoderAttention:
    """Test bidirectional attention."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=512,
        )
    
    def test_forward_shape(self, config):
        """Test output shape."""
        attn = Qwen3EncoderAttention(config)
        x = torch.randn(2, 10, 256)
        
        out, _ = attn(x)
        assert out.shape == x.shape
    
    def test_bidirectional_no_causal_mask(self, config):
        """Test that attention is truly bidirectional."""
        attn = Qwen3EncoderAttention(config)
        attn.eval()
        
        x = torch.randn(1, 5, 256)
        _, attn_weights = attn(x, output_attentions=True)
        
        # In bidirectional attention, all positions attend to all positions
        # Check that attention weights are non-zero for all positions
        assert attn_weights is not None
        assert attn_weights.shape == (1, 8, 5, 5)
        
        # All positions should have non-trivial attention
        assert (attn_weights > 0.01).any(dim=-1).all()
    
    def test_padding_mask(self, config):
        """Test that padding mask works."""
        attn = Qwen3EncoderAttention(config)
        attn.eval()
        
        x = torch.randn(2, 5, 256)
        
        # Mask last 2 positions in first batch, last 1 in second
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ], dtype=torch.float32)
        
        # Convert to attention mask format
        expanded_mask = attention_mask[:, None, None, :]
        expanded_mask = (1.0 - expanded_mask) * torch.finfo(torch.float32).min
        
        _, attn_weights = attn(x, attention_mask=expanded_mask, output_attentions=True)
        
        # Attention to masked positions should be ~0
        assert attn_weights[0, :, :, 3:].max() < 1e-5
        assert attn_weights[1, :, :, 4:].max() < 1e-5


class TestQwen3MLP:
    """Test MLP implementation."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            hidden_size=256,
            intermediate_size=512,
        )
    
    def test_forward_shape(self, config):
        """Test output shape."""
        mlp = Qwen3MLP(config)
        x = torch.randn(2, 10, 256)
        out = mlp(x)
        assert out.shape == x.shape


class TestQwen3EncoderLayer:
    """Test encoder layer."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=512,
        )
    
    def test_forward_shape(self, config):
        """Test output shape."""
        layer = Qwen3EncoderLayer(config)
        x = torch.randn(2, 10, 256)
        
        out, _ = layer(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self, config):
        """Test that residual connections work."""
        layer = Qwen3EncoderLayer(config)
        
        # Initialize to small weights to make residual dominant
        for p in layer.parameters():
            p.data.fill_(0.001)
        
        x = torch.randn(2, 10, 256)
        out, _ = layer(x)
        
        # Output should be close to input (residual connection)
        assert torch.allclose(out, x, atol=0.5)


class TestQwen3Encoder:
    """Test full encoder."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=4,
            intermediate_size=512,
        )
    
    @pytest.fixture
    def encoder(self, config):
        return Qwen3Encoder(config)
    
    def test_forward_with_input_ids(self, encoder):
        """Test forward pass with input IDs."""
        input_ids = torch.randint(0, 1000, (2, 20))
        
        outputs = encoder(input_ids)
        
        assert outputs.last_hidden_state.shape == (2, 20, 256)
    
    def test_forward_with_inputs_embeds(self, encoder):
        """Test forward pass with pre-computed embeddings."""
        inputs_embeds = torch.randn(2, 20, 256)
        
        outputs = encoder(inputs_embeds=inputs_embeds)
        
        assert outputs.last_hidden_state.shape == (2, 20, 256)
    
    def test_forward_with_attention_mask(self, encoder):
        """Test forward pass with padding mask."""
        input_ids = torch.randint(0, 1000, (2, 20))
        attention_mask = torch.ones(2, 20)
        attention_mask[0, 15:] = 0  # Pad last 5 positions
        
        outputs = encoder(input_ids, attention_mask=attention_mask)
        
        assert outputs.last_hidden_state.shape == (2, 20, 256)
    
    def test_output_hidden_states(self, encoder):
        """Test returning all hidden states."""
        input_ids = torch.randint(0, 1000, (2, 20))
        
        outputs = encoder(input_ids, output_hidden_states=True)
        
        assert outputs.hidden_states is not None
        # num_layers + 1 (final norm output)
        assert len(outputs.hidden_states) == 5
    
    def test_output_attentions(self, encoder):
        """Test returning attention weights."""
        input_ids = torch.randint(0, 1000, (2, 20))
        
        outputs = encoder(input_ids, output_attentions=True)
        
        assert outputs.attentions is not None
        assert len(outputs.attentions) == 4  # num_layers
    
    def test_gradient_flow(self, encoder):
        """Test that gradients flow through encoder."""
        input_ids = torch.randint(0, 1000, (2, 20))
        
        outputs = encoder(input_ids)
        loss = outputs.last_hidden_state.sum()
        loss.backward()
        
        # Check gradients exist
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_get_set_input_embeddings(self, encoder, config):
        """Test embedding getter/setter."""
        # Get
        embeddings = encoder.get_input_embeddings()
        assert isinstance(embeddings, nn.Embedding)
        
        # Set
        new_embeddings = nn.Embedding(2000, 256)
        encoder.set_input_embeddings(new_embeddings)
        assert encoder.embed_tokens is new_embeddings
    
    def test_return_dict_false(self, encoder):
        """Test tuple output when return_dict=False."""
        input_ids = torch.randint(0, 1000, (2, 20))
        
        outputs = encoder(input_ids, return_dict=False)
        
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3  # (hidden_states, all_hidden, all_attn)


class TestEncoderIntegration:
    """Integration tests for encoder."""
    
    def test_batch_independence(self):
        """Test that batches are processed independently."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        encoder = Qwen3Encoder(config)
        encoder.eval()
        
        # Process single sample
        input_ids = torch.randint(0, 1000, (1, 10))
        single_output = encoder(input_ids).last_hidden_state
        
        # Process as batch
        batch_input = input_ids.expand(3, -1)
        batch_output = encoder(batch_input).last_hidden_state
        
        # All batch outputs should be identical
        assert torch.allclose(single_output, batch_output[0:1], atol=1e-5)
        assert torch.allclose(batch_output[0], batch_output[1], atol=1e-5)
        assert torch.allclose(batch_output[1], batch_output[2], atol=1e-5)
    
    def test_different_sequence_lengths(self):
        """Test encoder handles different sequence lengths."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        encoder = Qwen3Encoder(config)
        encoder.eval()
        
        for seq_len in [5, 50, 200]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            outputs = encoder(input_ids)
            assert outputs.last_hidden_state.shape == (2, seq_len, 64)
```

---

## Acceptance Criteria

1. **RMSNorm**: Matches Qwen3 normalization behavior
2. **RoPE**: Correctly applies rotary position embeddings
3. **GQA**: Grouped query attention with correct head expansion
4. **QK-Norm**: Applied before attention computation
5. **Bidirectional**: No causal mask applied
6. **Padding Mask**: Correctly masks padded positions
7. **Pre-norm**: Layer norm before attention and MLP
8. **Gated MLP**: SiLU activation with gate projection
9. **Residual**: Skip connections work correctly
10. **Gradient Flow**: Gradients propagate through all layers
11. **Output Shapes**: All outputs have correct dimensions
12. **Unit Tests**: All tests pass with >95% coverage

---

## Performance Considerations

1. **Flash Attention**: Consider adding Flash Attention 2 support for efficiency
2. **Memory**: Implement gradient checkpointing for long sequences
3. **Compilation**: Test with `torch.compile()` for speedup
4. **Mixed Precision**: Ensure BF16/FP16 compatibility

---

## Notes for Developer

1. **Verify Qwen3 Source**: Before implementing, check the latest `modeling_qwen3.py` for any updates to attention or normalization.

2. **QK-Norm Position**: Qwen3 applies QK-norm BEFORE RoPE. Verify this is correct.

3. **Layer Types**: Qwen3 may have different layer types (sliding vs full attention). For the encoder, we may want all layers to use full bidirectional attention.

4. **Sliding Window**: Consider whether sliding window attention makes sense for the encoder. T5Gemma 2 may provide guidance.

5. **Weight Naming**: Use weight names that will map cleanly to Qwen3 checkpoint for easy loading in Story 06.

---

## Next Story

After completing this story, proceed to **Story 04: Merged Attention Decoder Implementation**.
