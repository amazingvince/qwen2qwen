"""Qwen3 Encoder-Decoder model implementation."""

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .encoder_only import (
    Qwen3EncoderConfig,
    Qwen3EncoderPooler,
    Qwen3EncoderPoolerOutput,
    Qwen3StandaloneEncoderModel,
)
from .modeling_qwen3_decoder import (
    Qwen3Decoder,
    Qwen3DecoderLayer,
    Qwen3DecoderOutput,
    Qwen3DecoderPreTrainedModel,
    Qwen3MergedAttention,
)
from .modeling_qwen3_encdec import (
    Qwen3ForSeq2SeqLM,
    Qwen3Seq2SeqModel,
    Qwen3Seq2SeqModelOutput,
    Qwen3Seq2SeqPreTrainedModel,
)
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
from .weight_initialization import (
    Qwen3WeightMapper,
    initialize_from_qwen3,
    verify_gradient_flow,
    verify_weight_initialization,
)

__version__ = "0.1.1"

__all__ = [
    # Configuration
    "Qwen3EncoderDecoderConfig",
    # Tokenization
    "Qwen3EncoderDecoderTokenizer",
    "SENTINEL_TOKEN_TEMPLATE",
    "create_sentinel_sequence",
    "apply_sentinel_corruption",
    # Seq2Seq - Models
    "Qwen3ForSeq2SeqLM",
    "Qwen3Seq2SeqModel",
    "Qwen3Seq2SeqPreTrainedModel",
    "Qwen3Seq2SeqModelOutput",
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
    # Decoder - Models
    "Qwen3Decoder",
    "Qwen3DecoderPreTrainedModel",
    "Qwen3DecoderOutput",
    # Decoder - Layers
    "Qwen3DecoderLayer",
    "Qwen3MergedAttention",
    # Weight Initialization
    "Qwen3WeightMapper",
    "initialize_from_qwen3",
    "verify_weight_initialization",
    "verify_gradient_flow",
    # Standalone Encoder (for embedding tasks)
    "Qwen3EncoderConfig",
    "Qwen3EncoderPooler",
    "Qwen3EncoderPoolerOutput",
    "Qwen3StandaloneEncoderModel",
]
