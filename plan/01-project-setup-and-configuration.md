# Story 01: Project Setup & Configuration Class

## Overview

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-001 |
| **Title** | Project Setup & Configuration Class |
| **Priority** | P0 - Critical Path |
| **Estimated Effort** | 2-3 days |
| **Dependencies** | None (first story) |
| **Deliverables** | Project structure, `Qwen3EncoderDecoderConfig` class, unit tests |

---

## Objective

Set up the project structure and implement the `Qwen3EncoderDecoderConfig` configuration class that will define all hyperparameters and architectural settings for the Qwen3 Encoder-Decoder model. This configuration class follows HuggingFace's `PretrainedConfig` pattern and will be used by all subsequent model components.

---

## Background & Context

### Why This Matters
The configuration class is the foundation of any HuggingFace model. It:
- Defines all model hyperparameters in a single, serializable location
- Enables `from_pretrained()` and `save_pretrained()` functionality
- Allows model architecture to be reconstructed from a JSON file
- Provides validation and default values for all parameters

### Reference Implementations
Study these before implementation:
1. **Qwen3Config**: `transformers/models/qwen3/configuration_qwen3.py`
2. **T5Config**: `transformers/models/t5/configuration_t5.py` (for encoder-decoder patterns)
3. **T5Gemma2 Config**: Check HuggingFace Hub for `google/t5gemma-2-*` config.json

---

## Technical Requirements

### 1. Project Directory Structure

Create the following directory structure:

```
qwen3_encdec/
├── __init__.py
├── configuration_qwen3_encdec.py      # This story
├── modeling_qwen3_encoder.py          # Story 03
├── modeling_qwen3_decoder.py          # Story 04
├── modeling_qwen3_encdec.py           # Story 05
├── tokenization_qwen3_encdec.py       # Story 02
├── tests/
│   ├── __init__.py
│   ├── test_configuration.py          # This story
│   ├── test_tokenization.py           # Story 02
│   ├── test_encoder.py                # Story 03
│   ├── test_decoder.py                # Story 04
│   ├── test_model.py                  # Story 05
│   └── test_weight_init.py            # Story 06
├── scripts/
│   ├── initialize_from_qwen3.py       # Story 06
│   ├── train.py                       # Story 10
│   └── extract_encoder.py             # Story 11
└── configs/
    └── default_config.json
```

### 2. Configuration Class Specification

#### File: `configuration_qwen3_encdec.py`

```python
"""Configuration for Qwen3 Encoder-Decoder model."""

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
        
        # Encoder-Decoder Specific Parameters
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether to tie encoder, decoder input, and output embeddings.
            Following T5Gemma 2, this provides ~10.5% parameter savings.
        use_merged_attention (`bool`, *optional*, defaults to True):
            Whether to use merged self/cross attention in the decoder.
            This follows T5Gemma 2's architecture for efficient parameter reuse.
        is_encoder_decoder (`bool`, *optional*, defaults to True):
            Flag indicating this is an encoder-decoder model.
        
        # Sentinel Token Parameters
        num_sentinel_tokens (`int`, *optional*, defaults to 100):
            Number of sentinel tokens to add for UL2 span corruption training.
        sentinel_token_start_id (`int`, *optional*, defaults to 151936):
            The token ID where sentinel tokens begin.
        
        # Initialization Parameters
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
        intermediate_size: int = 2816,
        hidden_act: str = "silu",
        
        # Normalization
        rms_norm_eps: float = 1e-6,
        
        # Positional encoding
        rope_theta: float = 10000.0,
        rope_scaling: dict = None,
        max_position_embeddings: int = 32768,
        
        # Attention configuration
        attention_dropout: float = 0.0,
        sliding_window: int = 32768,
        layer_types: list = None,
        
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
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
        decoder_start_token_id: int = None,
        
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        
        self.rms_norm_eps = rms_norm_eps
        
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        
        self.tie_word_embeddings = tie_word_embeddings
        self.use_merged_attention = use_merged_attention
        
        self.num_sentinel_tokens = num_sentinel_tokens
        self.sentinel_token_start_id = sentinel_token_start_id
        
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # Compute derived attributes
        self.head_dim = hidden_size // num_attention_heads
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
    
    @property
    def original_vocab_size(self) -> int:
        """Returns the original Qwen3 vocabulary size without sentinels."""
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
    
    @classmethod
    def from_qwen3_config(cls, qwen3_config, **kwargs):
        """
        Create an encoder-decoder config from an existing Qwen3 config.
        
        Args:
            qwen3_config: A Qwen3Config instance or path to config.
            **kwargs: Override any parameters.
            
        Returns:
            Qwen3EncoderDecoderConfig instance.
        """
        if isinstance(qwen3_config, str):
            from transformers import AutoConfig
            qwen3_config = AutoConfig.from_pretrained(qwen3_config)
        
        # Extract relevant parameters from Qwen3 config
        config_dict = {
            "hidden_size": qwen3_config.hidden_size,
            "num_hidden_layers": qwen3_config.num_hidden_layers,
            "num_attention_heads": qwen3_config.num_attention_heads,
            "num_key_value_heads": qwen3_config.num_key_value_heads,
            "intermediate_size": qwen3_config.intermediate_size,
            "hidden_act": qwen3_config.hidden_act,
            "rms_norm_eps": qwen3_config.rms_norm_eps,
            "rope_theta": qwen3_config.rope_theta,
            "max_position_embeddings": qwen3_config.max_position_embeddings,
            "attention_dropout": getattr(qwen3_config, "attention_dropout", 0.0),
            "sliding_window": getattr(qwen3_config, "sliding_window", 32768),
            "layer_types": getattr(qwen3_config, "layer_types", None),
        }
        
        # Update vocab_size to include sentinels
        num_sentinels = kwargs.get("num_sentinel_tokens", 100)
        config_dict["vocab_size"] = qwen3_config.vocab_size + num_sentinels
        config_dict["sentinel_token_start_id"] = qwen3_config.vocab_size
        
        # Override with any provided kwargs
        config_dict.update(kwargs)
        
        return cls(**config_dict)
```

### 3. Validation Logic

Add the following validation method to the config class:

```python
    def validate(self):
        """Validate configuration parameters."""
        errors = []
        
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
        
        return True
```

---

## Unit Tests

#### File: `tests/test_configuration.py`

```python
"""Unit tests for Qwen3EncoderDecoderConfig."""

import json
import tempfile
from pathlib import Path

import pytest

from qwen3_encdec.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig


class TestQwen3EncoderDecoderConfig:
    """Test suite for configuration class."""
    
    def test_default_initialization(self):
        """Test that default config matches Qwen3-0.6B architecture."""
        config = Qwen3EncoderDecoderConfig()
        
        # Core architecture
        assert config.vocab_size == 152036  # 151936 + 100 sentinels
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 28
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 2816
        
        # Encoder-decoder specific
        assert config.is_encoder_decoder is True
        assert config.tie_word_embeddings is True
        assert config.use_merged_attention is True
        
        # Sentinel tokens
        assert config.num_sentinel_tokens == 100
        assert config.sentinel_token_start_id == 151936
        
        # Model type
        assert config.model_type == "qwen3_encdec"
    
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        config = Qwen3EncoderDecoderConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
        )
        
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.num_key_value_heads == 4
        assert config.head_dim == 64  # 768 / 12
    
    def test_original_vocab_size_property(self):
        """Test original_vocab_size property computation."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=152036,
            num_sentinel_tokens=100,
        )
        assert config.original_vocab_size == 151936
    
    def test_get_sentinel_token_id(self):
        """Test sentinel token ID retrieval."""
        config = Qwen3EncoderDecoderConfig(
            sentinel_token_start_id=151936,
            num_sentinel_tokens=100,
        )
        
        # First sentinel
        assert config.get_sentinel_token_id(0) == 151936
        
        # Last sentinel
        assert config.get_sentinel_token_id(99) == 152035
        
        # Out of range
        with pytest.raises(ValueError):
            config.get_sentinel_token_id(100)
        
        with pytest.raises(ValueError):
            config.get_sentinel_token_id(-1)
    
    def test_validation_gqa(self):
        """Test GQA validation."""
        # Valid: 16 heads, 8 KV heads (16 % 8 == 0)
        config = Qwen3EncoderDecoderConfig(
            num_attention_heads=16,
            num_key_value_heads=8,
        )
        config.validate()  # Should not raise
        
        # Invalid: 16 heads, 7 KV heads (16 % 7 != 0)
        config = Qwen3EncoderDecoderConfig(
            num_attention_heads=16,
            num_key_value_heads=7,
        )
        with pytest.raises(ValueError, match="divisible"):
            config.validate()
    
    def test_validation_hidden_size(self):
        """Test hidden_size validation."""
        # Invalid: hidden_size not divisible by num_heads
        config = Qwen3EncoderDecoderConfig(
            hidden_size=1000,
            num_attention_heads=16,
        )
        with pytest.raises(ValueError, match="divisible"):
            config.validate()
    
    def test_save_and_load(self):
        """Test config serialization and deserialization."""
        config = Qwen3EncoderDecoderConfig(
            hidden_size=512,
            num_hidden_layers=6,
            custom_param="test_value",  # Extra kwargs should be preserved
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            config.save_pretrained(save_path)
            
            # Check JSON file was created
            config_file = save_path / "config.json"
            assert config_file.exists()
            
            # Load and verify
            loaded_config = Qwen3EncoderDecoderConfig.from_pretrained(save_path)
            assert loaded_config.hidden_size == 512
            assert loaded_config.num_hidden_layers == 6
            assert loaded_config.model_type == "qwen3_encdec"
    
    def test_to_json_string(self):
        """Test JSON serialization."""
        config = Qwen3EncoderDecoderConfig()
        json_str = config.to_json_string()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "qwen3_encdec"
        assert parsed["hidden_size"] == 1024
    
    def test_from_qwen3_config(self):
        """Test creating config from Qwen3 config."""
        # Create a mock Qwen3 config
        class MockQwen3Config:
            hidden_size = 1024
            num_hidden_layers = 28
            num_attention_heads = 16
            num_key_value_heads = 8
            intermediate_size = 2816
            hidden_act = "silu"
            rms_norm_eps = 1e-6
            rope_theta = 10000.0
            max_position_embeddings = 32768
            vocab_size = 151936
        
        mock_config = MockQwen3Config()
        
        config = Qwen3EncoderDecoderConfig.from_qwen3_config(mock_config)
        
        # Should have extended vocab
        assert config.vocab_size == 152036
        assert config.sentinel_token_start_id == 151936
        
        # Should inherit other params
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 28
    
    def test_head_dim_computation(self):
        """Test that head_dim is correctly computed."""
        config = Qwen3EncoderDecoderConfig(
            hidden_size=1024,
            num_attention_heads=16,
        )
        assert config.head_dim == 64
        
        config = Qwen3EncoderDecoderConfig(
            hidden_size=768,
            num_attention_heads=12,
        )
        assert config.head_dim == 64
```

---

## Default Configuration File

#### File: `configs/default_config.json`

```json
{
  "model_type": "qwen3_encdec",
  "vocab_size": 152036,
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "intermediate_size": 2816,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "rope_scaling": null,
  "max_position_embeddings": 32768,
  "attention_dropout": 0.0,
  "sliding_window": 32768,
  "layer_types": null,
  "tie_word_embeddings": true,
  "use_merged_attention": true,
  "is_encoder_decoder": true,
  "num_sentinel_tokens": 100,
  "sentinel_token_start_id": 151936,
  "initializer_range": 0.02,
  "use_cache": true,
  "pad_token_id": null,
  "bos_token_id": null,
  "eos_token_id": null,
  "decoder_start_token_id": null,
  "architectures": ["Qwen3ForSeq2SeqLM"],
  "transformers_version": "4.40.0"
}
```

---

## Package Initialization

#### File: `__init__.py`

```python
"""Qwen3 Encoder-Decoder model implementation."""

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig

__version__ = "0.1.0"

__all__ = [
    "Qwen3EncoderDecoderConfig",
]
```

---

## Acceptance Criteria

1. **Project Structure**: All directories and placeholder files are created
2. **Config Class**: `Qwen3EncoderDecoderConfig` implements all required parameters
3. **Inheritance**: Config properly extends `PretrainedConfig`
4. **Serialization**: Config can be saved and loaded via `save_pretrained()`/`from_pretrained()`
5. **Validation**: Invalid configurations raise appropriate errors
6. **Sentinel Tokens**: `get_sentinel_token_id()` works correctly
7. **Factory Method**: `from_qwen3_config()` properly creates config from Qwen3
8. **Unit Tests**: All tests pass with >95% coverage of config class
9. **Documentation**: All parameters have docstrings

---

## Dependencies

```txt
# requirements.txt
transformers>=4.40.0
torch>=2.0.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

---

## Notes for Developer

1. **Study Qwen3Config first**: Run `python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('Qwen/Qwen3-0.6B'); print(c)"` to see actual values

2. **Check T5Gemma 2 config**: The T5Gemma config on HuggingFace Hub may have additional encoder-decoder specific parameters worth adopting

3. **Vocab size verification**: The Qwen3 tokenizer vocab size should be verified - the 151,936 value is from documentation but may differ

4. **Layer types**: Qwen3 may have specific layer type patterns (sliding vs full attention) that should be preserved

---

## Next Story

After completing this story, proceed to **Story 02: Tokenizer Extension with Sentinel Tokens**.
