"""Configuration for Qwen3 Encoder-Decoder model."""

from typing import Any, Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3EncoderDecoderConfig(PretrainedConfig):
    """
    Configuration class for Qwen3 Encoder-Decoder model.

    This configuration stores all hyperparameters needed to instantiate a
    Qwen3-based encoder-decoder model following the T5Gemma 2 architecture pattern.

    Args:
        vocab_size (`int`, *optional*, defaults to 152036):
            Vocabulary size of the model. This includes the original Qwen3 vocabulary
            (151,936 tokens) plus 100 sentinel tokens for UL2 training.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in both the encoder and decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for Grouped Query Attention (GQA).
            Qwen3 uses GQA with 16 query heads and 8 KV heads.
        intermediate_size (`int`, *optional*, defaults to 2816):
            Dimensionality of the MLP intermediate layer.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function in the MLP.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by RMSNorm layers.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            Dictionary containing RoPE scaling configuration.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length the model can handle.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention weights.
        sliding_window (`int`, *optional*, defaults to 32768):
            Sliding window attention size. Set equal to max_position_embeddings
            for full attention.
        layer_types (`list`, *optional*):
            List specifying the attention type for each layer.
            Options: "sliding_attention" or "full_attention".
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether to tie encoder, decoder input, and output embeddings.
            Following T5Gemma 2, this provides ~10.5% parameter savings.
        use_merged_attention (`bool`, *optional*, defaults to True):
            Whether to use merged self/cross attention in the decoder.
            This follows T5Gemma 2's architecture for efficient parameter reuse.
        is_encoder_decoder (`bool`, *optional*, defaults to True):
            Flag indicating this is an encoder-decoder model.
        num_sentinel_tokens (`int`, *optional*, defaults to 100):
            Number of sentinel tokens to add for UL2 span corruption training.
        sentinel_token_start_id (`int`, *optional*, defaults to 151936):
            The token ID where sentinel tokens begin.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        use_cache (`bool`, *optional*, defaults to True):
            Whether to use KV cache for generation.

    Example:
        ```python
        from qwen3_encdec import Qwen3EncoderDecoderConfig

        # Default configuration (matches Qwen3-0.6B architecture)
        config = Qwen3EncoderDecoderConfig()

        # Custom configuration
        config = Qwen3EncoderDecoderConfig(
            num_hidden_layers=24,
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=4,
        )
        ```
    """

    model_type = "qwen3_encdec"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Core architecture (from Qwen3-0.6B)
        vocab_size: int = 152036,  # 151936 + 100 sentinels
        hidden_size: int = 1024,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,  # If None, computed from hidden_size//num_heads
        intermediate_size: int = 3072,  # Qwen3-0.6B uses 3072
        hidden_act: str = "silu",
        # Normalization
        rms_norm_eps: float = 1e-6,
        # Positional encoding
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        # Attention configuration
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = 32768,
        layer_types: Optional[List[str]] = None,
        # Encoder-Decoder specific
        tie_word_embeddings: bool = True,
        use_merged_attention: bool = True,
        is_encoder_decoder: bool = True,
        # Sentinel tokens
        num_sentinel_tokens: int = 100,
        sentinel_token_start_id: int = 151936,
        # Initialization
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # Special tokens
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self._head_dim = head_dim  # Store explicit head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.rms_norm_eps = rms_norm_eps

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings

        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types

        self.use_merged_attention = use_merged_attention

        self.num_sentinel_tokens = num_sentinel_tokens
        self.sentinel_token_start_id = sentinel_token_start_id

        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Auto-validate configuration on construction
        self._validate()

    @property
    def head_dim(self) -> int:
        """Return the dimension of each attention head."""
        if self._head_dim is not None:
            return self._head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        """Return the number of query head groups per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def original_vocab_size(self) -> int:
        """Return the original Qwen3 vocabulary size without sentinels."""
        return self.vocab_size - self.num_sentinel_tokens

    def get_sentinel_token_id(self, index: int) -> int:
        """
        Get the token ID for a sentinel token.

        Args:
            index: Sentinel index (0 to num_sentinel_tokens - 1)

        Returns:
            The token ID for the sentinel token.

        Raises:
            ValueError: If index is out of range.
        """
        if not 0 <= index < self.num_sentinel_tokens:
            raise ValueError(
                f"Sentinel index must be between 0 and {self.num_sentinel_tokens - 1}, "
                f"got {index}"
            )
        return self.sentinel_token_start_id + index

    def _validate(self) -> None:
        """Validate configuration parameters. Raises ValueError if invalid."""
        errors: List[str] = []

        # Validate GQA configuration
        if self.num_attention_heads % self.num_key_value_heads != 0:
            errors.append(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )

        # Validate hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # Validate sentinel tokens
        if self.num_sentinel_tokens < 0:
            errors.append("num_sentinel_tokens must be non-negative")

        if self.vocab_size != self.sentinel_token_start_id + self.num_sentinel_tokens:
            errors.append(
                f"vocab_size ({self.vocab_size}) must equal "
                f"sentinel_token_start_id ({self.sentinel_token_start_id}) + "
                f"num_sentinel_tokens ({self.num_sentinel_tokens})"
            )

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if validation passes.

        Raises:
            ValueError: If validation fails.
        """
        self._validate()
        return True

    @classmethod
    def from_qwen3_config(
        cls,
        qwen3_config: Union[str, "PretrainedConfig"],
        num_sentinel_tokens: int = 100,
        **kwargs: Any,
    ) -> "Qwen3EncoderDecoderConfig":
        """
        Create an encoder-decoder config from an existing Qwen3 config.

        Args:
            qwen3_config: A Qwen3Config instance or path to config/model.
            num_sentinel_tokens: Number of sentinel tokens to add.
            **kwargs: Override any parameters.

        Returns:
            Qwen3EncoderDecoderConfig instance.

        Raises:
            ValueError: If qwen3_config is invalid or missing required attributes.
        """
        if isinstance(qwen3_config, str):
            from transformers import AutoConfig

            qwen3_config = AutoConfig.from_pretrained(qwen3_config)

        # Required attributes from Qwen3 config
        required_attrs = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "hidden_act",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "vocab_size",
        ]

        missing = [attr for attr in required_attrs if not hasattr(qwen3_config, attr)]
        if missing:
            raise ValueError(
                f"qwen3_config is missing required attributes: {missing}"
            )

        # Extract relevant parameters from Qwen3 config
        config_dict: Dict[str, Any] = {
            "hidden_size": qwen3_config.hidden_size,
            "num_hidden_layers": qwen3_config.num_hidden_layers,
            "num_attention_heads": qwen3_config.num_attention_heads,
            "num_key_value_heads": qwen3_config.num_key_value_heads,
            "head_dim": getattr(qwen3_config, "head_dim", None),  # Qwen3 may have explicit head_dim
            "intermediate_size": qwen3_config.intermediate_size,
            "hidden_act": qwen3_config.hidden_act,
            "rms_norm_eps": qwen3_config.rms_norm_eps,
            "rope_theta": qwen3_config.rope_theta,
            "max_position_embeddings": qwen3_config.max_position_embeddings,
            "attention_dropout": getattr(qwen3_config, "attention_dropout", 0.0),
            "sliding_window": getattr(qwen3_config, "sliding_window", 32768),
            "layer_types": getattr(qwen3_config, "layer_types", None),
        }

        # Handle rope_scaling if present
        if hasattr(qwen3_config, "rope_scaling") and qwen3_config.rope_scaling:
            config_dict["rope_scaling"] = qwen3_config.rope_scaling

        # Update vocab_size to include sentinels
        config_dict["vocab_size"] = qwen3_config.vocab_size + num_sentinel_tokens
        config_dict["sentinel_token_start_id"] = qwen3_config.vocab_size
        config_dict["num_sentinel_tokens"] = num_sentinel_tokens

        # Override with any provided kwargs
        config_dict.update(kwargs)

        return cls(**config_dict)

    def __repr__(self) -> str:
        """Return a string representation of the config."""
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"hidden_size={self.hidden_size}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"num_sentinel_tokens={self.num_sentinel_tokens})"
        )
