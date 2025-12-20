# Story 04: Merged Attention Decoder Implementation

## Overview

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-004 |
| **Title** | Merged Attention Decoder Implementation |
| **Priority** | P0 - Critical Path |
| **Estimated Effort** | 4-5 days |
| **Dependencies** | Story 01 (Configuration), Story 03 (Encoder components) |
| **Deliverables** | `Qwen3MergedAttentionDecoder`, merged attention layer, unit tests |

---

## Objective

Implement the decoder component with **merged self/cross attention** following the T5Gemma 2 architecture pattern. This is the most complex component of the encoder-decoder model.

---

## Background & Context

### What is Merged Attention?
Traditional encoder-decoder models have separate self-attention and cross-attention layers:
```
Decoder Layer:
  Self-Attention (Q, K, V from decoder hidden) 
  Cross-Attention (Q from decoder, K, V from encoder)
  MLP
```

Merged attention combines these into a single attention operation:
```
Decoder Layer (Merged):
  Merged Attention:
    Q: from decoder hidden only
    K: concat(decoder hidden, encoder output)
    V: concat(decoder hidden, encoder output)
  MLP
```

### Benefits of Merged Attention
1. **Parameter Efficiency**: Same Wq, Wk, Wv, Wo used for both self and cross attention
2. **Pretrained Weight Reuse**: Can directly use Qwen3's attention weights
3. **Simplified Architecture**: One attention op instead of two
4. **Proven Approach**: T5Gemma 2 validates this works

### Attention Mask Structure
The merged attention mask must handle:
- **Decoder-to-Decoder**: Causal (lower triangular)
- **Decoder-to-Encoder**: Full attention (no masking)

```
For decoder length m and encoder length n:

Mask shape: [m, m+n]

       |<-- decoder -->|<-- encoder -->|
       | d0  d1  ...dm | e0  e1  ...en |
  d0   | 1   0   ...0  | 1   1   ...1  |
  d1   | 1   1   ...0  | 1   1   ...1  |
  ...  |               |               |
  dm   | 1   1   ...1  | 1   1   ...1  |

Where 1 = attend, 0 = mask
```

---

## Technical Requirements

### 1. Merged Attention Class

#### File: `modeling_qwen3_decoder.py`

```python
"""Qwen3 Decoder with merged self/cross attention."""

import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_encoder import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3MLP,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.get_logger(__name__)


class Qwen3MergedAttention(nn.Module):
    """
    Merged self/cross attention for Qwen3 Decoder.
    
    This combines self-attention and cross-attention into a single operation:
    - Q comes from decoder hidden states only
    - K, V come from concatenation of [decoder hidden, encoder output]
    
    This allows reusing pretrained attention weights from Qwen3.
    
    Args:
        config: Model configuration.
        layer_idx: Layer index (for KV cache).
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
        
        # Projection layers (same weights for self and cross attention)
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
    
    def _compute_qkv(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_position_ids: torch.Tensor,
        encoder_position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V for merged attention.
        
        Args:
            decoder_hidden_states: [batch, dec_len, hidden_size]
            encoder_hidden_states: [batch, enc_len, hidden_size]
            decoder_position_ids: [batch, dec_len]
            encoder_position_ids: [batch, enc_len]
            
        Returns:
            Tuple of (query, key, value) tensors.
        """
        batch_size = decoder_hidden_states.shape[0]
        dec_len = decoder_hidden_states.shape[1]
        enc_len = encoder_hidden_states.shape[1]
        
        # Q from decoder only
        query_states = self.q_proj(decoder_hidden_states)
        query_states = query_states.view(
            batch_size, dec_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # K, V from concatenated [decoder, encoder]
        combined_hidden = torch.cat(
            [decoder_hidden_states, encoder_hidden_states], dim=1
        )
        
        key_states = self.k_proj(combined_hidden)
        value_states = self.v_proj(combined_hidden)
        
        key_states = key_states.view(
            batch_size, dec_len + enc_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, dec_len + enc_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply QK-Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Apply RoPE with appropriate positions
        # Q gets decoder positions
        dec_cos, dec_sin = self.rotary_emb(query_states, decoder_position_ids)
        query_states = self._apply_rope_single(query_states, dec_cos, dec_sin)
        
        # K gets separate positions for decoder and encoder portions
        # Decoder portion
        key_decoder = key_states[:, :, :dec_len, :]
        key_decoder = self._apply_rope_single(key_decoder, dec_cos, dec_sin)
        
        # Encoder portion (independent positions starting from 0)
        enc_cos, enc_sin = self.rotary_emb(key_states[:, :, dec_len:, :], encoder_position_ids)
        key_encoder = key_states[:, :, dec_len:, :]
        key_encoder = self._apply_rope_single(key_encoder, enc_cos, enc_sin)
        
        # Recombine key
        key_states = torch.cat([key_decoder, key_encoder], dim=2)
        
        return query_states, key_states, value_states
    
    def _apply_rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to a single tensor."""
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (rotated * sin)
    
    def _create_merged_attention_mask(
        self,
        decoder_length: int,
        encoder_length: int,
        decoder_attention_mask: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create merged attention mask for decoder.
        
        The mask has shape [batch, 1, dec_len, dec_len + enc_len] where:
        - Decoder-to-decoder: causal (lower triangular)
        - Decoder-to-encoder: full (all ones, respecting encoder padding)
        
        Args:
            decoder_length: Length of decoder sequence.
            encoder_length: Length of encoder sequence.
            decoder_attention_mask: Padding mask for decoder [batch, dec_len].
            encoder_attention_mask: Padding mask for encoder [batch, enc_len].
            dtype: Tensor dtype.
            device: Tensor device.
            
        Returns:
            Combined attention mask [batch, 1, dec_len, dec_len + enc_len].
        """
        # Causal mask for decoder-to-decoder
        causal_mask = torch.triu(
            torch.ones(decoder_length, decoder_length, device=device, dtype=dtype),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, torch.finfo(dtype).min)
        
        # Full attention for decoder-to-encoder
        cross_mask = torch.zeros(
            decoder_length, encoder_length, device=device, dtype=dtype
        )
        
        # Combine masks
        combined_mask = torch.cat([causal_mask, cross_mask], dim=1)
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, dec, dec+enc]
        
        # Apply padding masks if provided
        if encoder_attention_mask is not None:
            # Expand encoder mask for cross-attention portion
            enc_pad_mask = (1.0 - encoder_attention_mask.to(dtype)) * torch.finfo(dtype).min
            enc_pad_mask = enc_pad_mask[:, None, None, :]  # [batch, 1, 1, enc_len]
            
            # Apply to encoder portion of combined mask
            batch_size = encoder_attention_mask.shape[0]
            combined_mask = combined_mask.expand(batch_size, -1, -1, -1).clone()
            combined_mask[:, :, :, decoder_length:] += enc_pad_mask
        
        if decoder_attention_mask is not None:
            # Apply decoder padding mask
            dec_pad_mask = (1.0 - decoder_attention_mask.to(dtype)) * torch.finfo(dtype).min
            dec_pad_mask = dec_pad_mask[:, None, None, :]  # [batch, 1, 1, dec_len]
            
            if combined_mask.shape[0] == 1:
                batch_size = decoder_attention_mask.shape[0]
                combined_mask = combined_mask.expand(batch_size, -1, -1, -1).clone()
            
            combined_mask[:, :, :, :decoder_length] += dec_pad_mask
        
        return combined_mask
    
    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.Tensor] = None,
        encoder_position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for merged attention.
        
        Args:
            decoder_hidden_states: [batch, dec_len, hidden_size]
            encoder_hidden_states: [batch, enc_len, hidden_size]
            decoder_attention_mask: Padding mask for decoder [batch, dec_len]
            encoder_attention_mask: Padding mask for encoder [batch, enc_len]
            decoder_position_ids: Position IDs for decoder [batch, dec_len]
            encoder_position_ids: Position IDs for encoder [batch, enc_len]
            past_key_value: Cached KV states for incremental decoding
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights, past_key_value).
        """
        batch_size = decoder_hidden_states.shape[0]
        dec_len = decoder_hidden_states.shape[1]
        enc_len = encoder_hidden_states.shape[1]
        device = decoder_hidden_states.device
        dtype = decoder_hidden_states.dtype
        
        # Create position IDs if not provided
        if decoder_position_ids is None:
            decoder_position_ids = torch.arange(dec_len, device=device)
            decoder_position_ids = decoder_position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if encoder_position_ids is None:
            encoder_position_ids = torch.arange(enc_len, device=device)
            encoder_position_ids = encoder_position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Compute Q, K, V
        query_states, key_states, value_states = self._compute_qkv(
            decoder_hidden_states,
            encoder_hidden_states,
            decoder_position_ids,
            encoder_position_ids,
        )
        
        # Handle KV cache for incremental decoding
        if past_key_value is not None:
            # Append to cache
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # Repeat KV heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        # Apply merged attention mask
        attention_mask = self._create_merged_attention_mask(
            dec_len, enc_len,
            decoder_attention_mask, encoder_attention_mask,
            dtype, device,
        )
        attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, dec_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


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
    
    def __init__(self, config: Qwen3EncoderDecoderConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-norm before attention
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Merged self/cross attention
        self.merged_attn = Qwen3MergedAttention(config, layer_idx)
        
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
        decoder_position_ids: Optional[torch.Tensor] = None,
        encoder_position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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
        attn_output, attn_weights, present_key_value = self.merged_attn(
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


class Qwen3Decoder(PreTrainedModel):
    """
    Qwen3-based decoder with merged self/cross attention.
    
    Args:
        config: Model configuration.
    """
    
    config_class = Qwen3EncoderDecoderConfig
    
    def __init__(self, config: Qwen3EncoderDecoderConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings (will be tied with encoder)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i)
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
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, "Qwen3DecoderOutput"]:
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
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, dec_len, _ = inputs_embeds.shape
        enc_len = encoder_hidden_states.shape[1]
        device = inputs_embeds.device
        
        # Prepare position IDs
        if position_ids is None:
            if past_key_values is not None:
                # Incremental decoding: use appropriate offset
                past_length = past_key_values[0][0].shape[2] - enc_len
                position_ids = torch.arange(
                    past_length, past_length + dec_len, device=device
                )
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
                all_hidden_states += (hidden_states,)
            
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
                next_decoder_cache += (present_key_value,)
            
            if output_attentions:
                all_attentions += (attn_weights,)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if return_dict:
            return Qwen3DecoderOutput(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        
        return (hidden_states, next_decoder_cache, all_hidden_states, all_attentions)


# Output dataclass
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput


@dataclass
class Qwen3DecoderOutput(ModelOutput):
    """
    Output type for Qwen3Decoder.
    
    Args:
        last_hidden_state: Final layer hidden states [batch, dec_len, hidden_size].
        past_key_values: Cached KV states for incremental decoding.
        hidden_states: All hidden states if output_hidden_states=True.
        attentions: All attention weights if output_attentions=True.
    """
    
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
```

---

## Unit Tests

#### File: `tests/test_decoder.py`

```python
"""Unit tests for Qwen3 Decoder with merged attention."""

import pytest
import torch
import torch.nn as nn

from qwen3_encdec.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from qwen3_encdec.modeling_qwen3_decoder import (
    Qwen3Decoder,
    Qwen3DecoderLayer,
    Qwen3MergedAttention,
)


class TestQwen3MergedAttention:
    """Test merged attention implementation."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=512,
        )
    
    @pytest.fixture
    def attn(self, config):
        return Qwen3MergedAttention(config)
    
    def test_forward_shape(self, attn):
        """Test output shape."""
        decoder_hidden = torch.randn(2, 10, 256)
        encoder_hidden = torch.randn(2, 20, 256)
        
        output, _, _ = attn(decoder_hidden, encoder_hidden)
        
        assert output.shape == decoder_hidden.shape
    
    def test_attention_mask_shape(self, attn):
        """Test attention mask creation."""
        decoder_hidden = torch.randn(2, 10, 256)
        encoder_hidden = torch.randn(2, 20, 256)
        
        _, attn_weights, _ = attn(
            decoder_hidden, encoder_hidden, output_attentions=True
        )
        
        # Shape should be [batch, heads, dec_len, dec_len + enc_len]
        assert attn_weights.shape == (2, 8, 10, 30)
    
    def test_causal_masking_decoder_to_decoder(self, attn):
        """Test that decoder-to-decoder attention is causal."""
        attn.eval()
        decoder_hidden = torch.randn(1, 5, 256)
        encoder_hidden = torch.randn(1, 3, 256)
        
        _, attn_weights, _ = attn(
            decoder_hidden, encoder_hidden, output_attentions=True
        )
        
        # Decoder portion is [0:5] in the K dimension
        decoder_attn = attn_weights[:, :, :, :5]  # [1, 8, 5, 5]
        
        # Upper triangle should be ~0 (masked)
        for i in range(5):
            for j in range(i + 1, 5):
                assert decoder_attn[:, :, i, j].abs().max() < 1e-5
    
    def test_full_attention_decoder_to_encoder(self, attn):
        """Test that decoder-to-encoder attention is full."""
        attn.eval()
        decoder_hidden = torch.randn(1, 5, 256)
        encoder_hidden = torch.randn(1, 3, 256)
        
        _, attn_weights, _ = attn(
            decoder_hidden, encoder_hidden, output_attentions=True
        )
        
        # Encoder portion is [5:8] in the K dimension
        encoder_attn = attn_weights[:, :, :, 5:]  # [1, 8, 5, 3]
        
        # All positions should have some attention
        assert (encoder_attn > 0.01).any(dim=-1).all()
    
    def test_kv_cache(self, attn):
        """Test KV cache for incremental decoding."""
        encoder_hidden = torch.randn(1, 10, 256)
        
        # First token
        decoder_hidden_1 = torch.randn(1, 1, 256)
        output_1, _, cache_1 = attn(
            decoder_hidden_1, encoder_hidden, use_cache=True
        )
        
        # Second token (with cache)
        decoder_hidden_2 = torch.randn(1, 1, 256)
        decoder_position_ids = torch.tensor([[1]])
        output_2, _, cache_2 = attn(
            decoder_hidden_2, encoder_hidden,
            decoder_position_ids=decoder_position_ids,
            past_key_value=cache_1,
            use_cache=True,
        )
        
        assert output_1.shape == (1, 1, 256)
        assert output_2.shape == (1, 1, 256)
        assert cache_2[0].shape[2] > cache_1[0].shape[2]  # Cache grew
    
    def test_encoder_padding_mask(self, attn):
        """Test encoder padding mask is applied."""
        attn.eval()
        decoder_hidden = torch.randn(2, 3, 256)
        encoder_hidden = torch.randn(2, 5, 256)
        
        # Mask last 2 positions in encoder
        encoder_attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ], dtype=torch.float32)
        
        _, attn_weights, _ = attn(
            decoder_hidden, encoder_hidden,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=True,
        )
        
        # Encoder portion starts at position 3 (after decoder)
        encoder_attn = attn_weights[:, :, :, 3:]
        
        # Masked positions should have ~0 attention
        assert encoder_attn[0, :, :, 3:].max() < 1e-5
        assert encoder_attn[1, :, :, 4:].max() < 1e-5


class TestQwen3DecoderLayer:
    """Test decoder layer."""
    
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
        layer = Qwen3DecoderLayer(config)
        
        decoder_hidden = torch.randn(2, 10, 256)
        encoder_hidden = torch.randn(2, 20, 256)
        
        output, _, _ = layer(decoder_hidden, encoder_hidden)
        
        assert output.shape == decoder_hidden.shape
    
    def test_residual_connection(self, config):
        """Test residual connections work."""
        layer = Qwen3DecoderLayer(config)
        
        # Small weights to make residual dominant
        for p in layer.parameters():
            p.data.fill_(0.001)
        
        decoder_hidden = torch.randn(2, 10, 256)
        encoder_hidden = torch.randn(2, 20, 256)
        
        output, _, _ = layer(decoder_hidden, encoder_hidden)
        
        assert torch.allclose(output, decoder_hidden, atol=1.0)


class TestQwen3Decoder:
    """Test full decoder."""
    
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
    def decoder(self, config):
        return Qwen3Decoder(config)
    
    def test_forward_basic(self, decoder):
        """Test basic forward pass."""
        input_ids = torch.randint(0, 1000, (2, 10))
        encoder_hidden = torch.randn(2, 20, 256)
        
        outputs = decoder(input_ids, encoder_hidden_states=encoder_hidden)
        
        assert outputs.last_hidden_state.shape == (2, 10, 256)
    
    def test_forward_without_encoder_raises(self, decoder):
        """Test that missing encoder raises error."""
        input_ids = torch.randint(0, 1000, (2, 10))
        
        with pytest.raises(ValueError, match="encoder_hidden_states"):
            decoder(input_ids)
    
    def test_output_hidden_states(self, decoder):
        """Test returning all hidden states."""
        input_ids = torch.randint(0, 1000, (2, 10))
        encoder_hidden = torch.randn(2, 20, 256)
        
        outputs = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            output_hidden_states=True,
        )
        
        assert outputs.hidden_states is not None
        assert len(outputs.hidden_states) == 5  # 4 layers + final
    
    def test_output_attentions(self, decoder):
        """Test returning attention weights."""
        input_ids = torch.randint(0, 1000, (2, 10))
        encoder_hidden = torch.randn(2, 20, 256)
        
        outputs = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            output_attentions=True,
        )
        
        assert outputs.attentions is not None
        assert len(outputs.attentions) == 4
    
    def test_kv_cache(self, decoder):
        """Test KV cache for generation."""
        encoder_hidden = torch.randn(1, 10, 256)
        
        # First token
        input_ids = torch.randint(0, 1000, (1, 1))
        outputs_1 = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            use_cache=True,
        )
        
        assert outputs_1.past_key_values is not None
        
        # Second token
        input_ids = torch.randint(0, 1000, (1, 1))
        outputs_2 = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            past_key_values=outputs_1.past_key_values,
            use_cache=True,
        )
        
        assert outputs_2.past_key_values is not None
    
    def test_gradient_flow(self, decoder):
        """Test gradients flow through decoder."""
        input_ids = torch.randint(0, 1000, (2, 10))
        encoder_hidden = torch.randn(2, 20, 256, requires_grad=True)
        
        outputs = decoder(input_ids, encoder_hidden_states=encoder_hidden)
        loss = outputs.last_hidden_state.sum()
        loss.backward()
        
        # Gradients should flow to encoder
        assert encoder_hidden.grad is not None
        assert (encoder_hidden.grad != 0).any()
        
        # Gradients should exist for decoder params
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestDecoderIntegration:
    """Integration tests for decoder."""
    
    def test_autoregressive_generation_consistency(self):
        """Test that autoregressive generation matches batch processing."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        decoder = Qwen3Decoder(config)
        decoder.eval()
        
        encoder_hidden = torch.randn(1, 5, 64)
        input_ids = torch.randint(0, 100, (1, 4))
        
        # Full forward pass
        with torch.no_grad():
            full_outputs = decoder(input_ids, encoder_hidden_states=encoder_hidden)
            full_hidden = full_outputs.last_hidden_state
        
        # Autoregressive with cache
        with torch.no_grad():
            cache = None
            auto_hidden = []
            
            for i in range(4):
                step_input = input_ids[:, i:i+1]
                position_ids = torch.tensor([[i]])
                
                outputs = decoder(
                    step_input,
                    encoder_hidden_states=encoder_hidden,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                
                cache = outputs.past_key_values
                auto_hidden.append(outputs.last_hidden_state)
            
            auto_hidden = torch.cat(auto_hidden, dim=1)
        
        # Should be close (may have small numerical differences)
        assert torch.allclose(full_hidden, auto_hidden, atol=1e-4)
```

---

## Acceptance Criteria

1. **Merged Attention**: Single attention op handles both self and cross attention
2. **Causal Masking**: Decoder-to-decoder attention is causal
3. **Full Cross Attention**: Decoder-to-encoder attention has no masking
4. **RoPE Positions**: Encoder and decoder have independent position IDs
5. **QK-Norm**: Applied correctly in merged attention
6. **GQA**: Key-value head expansion works
7. **Padding Masks**: Both encoder and decoder padding respected
8. **KV Cache**: Incremental decoding works correctly
9. **Gradient Flow**: Gradients propagate from decoder to encoder
10. **Unit Tests**: All tests pass with >95% coverage

---

## Performance Considerations

1. **Memory**: Merged KV is larger than separate - monitor memory usage
2. **Flash Attention**: May need custom implementation for merged attention
3. **Cache Size**: Cache includes both decoder and encoder KV
4. **Compilation**: Test torch.compile compatibility

---

## Notes for Developer

1. **RoPE Position Handling**: The encoder and decoder have independent position embeddings. Verify this matches T5Gemma 2's approach.

2. **KV Cache Structure**: The cache format may need adjustment for generation. Test with `model.generate()` after Story 05.

3. **Weight Names**: Use names compatible with Qwen3 for easy weight loading in Story 06.

4. **Attention Mask Broadcasting**: Pay attention to batch dimension handling in mask creation.

5. **Sliding Window**: If Qwen3 uses sliding window attention, consider whether it applies to the merged attention case.

---

## Next Story

After completing this story, proceed to **Story 05: Qwen3ForSeq2SeqLM Combined Model**.
