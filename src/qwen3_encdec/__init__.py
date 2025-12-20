"""Qwen3 Encoder-Decoder model implementation."""

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_encoder import (
    Qwen3Encoder,
    Qwen3EncoderAttention,
    Qwen3EncoderLayer,
    Qwen3EncoderModel,
    Qwen3EncoderOutput,
    Qwen3EncoderPreTrainedModel,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from .tokenization_qwen3_encdec import (
    SENTINEL_TOKEN_TEMPLATE,
    Qwen3EncoderDecoderTokenizer,
    apply_sentinel_corruption,
    create_sentinel_sequence,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "Qwen3EncoderDecoderConfig",
    # Tokenization
    "Qwen3EncoderDecoderTokenizer",
    "SENTINEL_TOKEN_TEMPLATE",
    "create_sentinel_sequence",
    "apply_sentinel_corruption",
    # Encoder - Models
    "Qwen3Encoder",
    "Qwen3EncoderModel",
    "Qwen3EncoderPreTrainedModel",
    "Qwen3EncoderOutput",
    # Encoder - Layers
    "Qwen3EncoderLayer",
    "Qwen3EncoderAttention",
    "Qwen3MLP",
    # Encoder - Building Blocks
    "Qwen3RMSNorm",
    "Qwen3RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
]
