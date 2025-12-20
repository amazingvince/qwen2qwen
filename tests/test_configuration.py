"""Unit tests for Qwen3EncoderDecoderConfig."""

import json
import tempfile
from pathlib import Path

import pytest

from qwen3_encdec.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig


class TestQwen3EncoderDecoderConfigDefaults:
    """Test suite for default configuration values."""

    def test_default_initialization(self):
        """Test that default config matches Qwen3-0.6B architecture."""
        config = Qwen3EncoderDecoderConfig()

        # Core architecture
        assert config.vocab_size == 152036  # 151936 + 100 sentinels
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 28
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 3072  # Qwen3-0.6B uses 3072
        assert config.hidden_act == "silu"

        # Normalization
        assert config.rms_norm_eps == 1e-6

        # Positional encoding
        assert config.rope_theta == 10000.0
        assert config.rope_scaling is None
        assert config.max_position_embeddings == 32768

        # Attention
        assert config.attention_dropout == 0.0
        assert config.sliding_window == 32768

        # Encoder-decoder specific
        assert config.is_encoder_decoder is True
        assert config.tie_word_embeddings is True
        assert config.use_merged_attention is True

        # Sentinel tokens
        assert config.num_sentinel_tokens == 100
        assert config.sentinel_token_start_id == 151936

        # Model type
        assert config.model_type == "qwen3_encdec"


class TestQwen3EncoderDecoderConfigCustom:
    """Test suite for custom configuration values."""

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        config = Qwen3EncoderDecoderConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            vocab_size=32100,
            num_sentinel_tokens=100,
            sentinel_token_start_id=32000,
        )

        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.num_key_value_heads == 4
        assert config.vocab_size == 32100

    def test_custom_sentinel_tokens(self):
        """Test custom sentinel token configuration."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=1000 + 50,
            num_sentinel_tokens=50,
            sentinel_token_start_id=1000,
        )

        assert config.num_sentinel_tokens == 50
        assert config.sentinel_token_start_id == 1000
        assert config.vocab_size == 1050


class TestQwen3EncoderDecoderConfigProperties:
    """Test suite for configuration properties."""

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
            num_key_value_heads=4,
            vocab_size=152036,
            sentinel_token_start_id=151936,
        )
        assert config.head_dim == 64

    def test_num_key_value_groups(self):
        """Test that num_key_value_groups is correctly computed."""
        config = Qwen3EncoderDecoderConfig(
            num_attention_heads=16,
            num_key_value_heads=8,
        )
        assert config.num_key_value_groups == 2

        config = Qwen3EncoderDecoderConfig(
            num_attention_heads=12,
            num_key_value_heads=4,
            hidden_size=768,
            vocab_size=152036,
            sentinel_token_start_id=151936,
        )
        assert config.num_key_value_groups == 3

    def test_original_vocab_size_property(self):
        """Test original_vocab_size property computation."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=152036,
            num_sentinel_tokens=100,
        )
        assert config.original_vocab_size == 151936


class TestSentinelTokens:
    """Test suite for sentinel token methods."""

    def test_get_sentinel_token_id_valid(self):
        """Test sentinel token ID retrieval with valid indices."""
        config = Qwen3EncoderDecoderConfig(
            sentinel_token_start_id=151936,
            num_sentinel_tokens=100,
        )

        # First sentinel
        assert config.get_sentinel_token_id(0) == 151936

        # Middle sentinel
        assert config.get_sentinel_token_id(50) == 151986

        # Last sentinel
        assert config.get_sentinel_token_id(99) == 152035

    def test_get_sentinel_token_id_invalid_high(self):
        """Test sentinel token ID with index too high."""
        config = Qwen3EncoderDecoderConfig(
            sentinel_token_start_id=151936,
            num_sentinel_tokens=100,
        )

        with pytest.raises(ValueError, match="must be between 0 and 99"):
            config.get_sentinel_token_id(100)

    def test_get_sentinel_token_id_invalid_negative(self):
        """Test sentinel token ID with negative index."""
        config = Qwen3EncoderDecoderConfig()

        with pytest.raises(ValueError, match="must be between 0 and"):
            config.get_sentinel_token_id(-1)


class TestValidation:
    """Test suite for configuration validation."""

    def test_validation_gqa_valid(self):
        """Test GQA validation with valid configuration."""
        # Valid: 16 heads, 8 KV heads (16 % 8 == 0)
        config = Qwen3EncoderDecoderConfig(
            num_attention_heads=16,
            num_key_value_heads=8,
        )
        assert config.validate() is True

    def test_validation_gqa_invalid(self):
        """Test GQA validation with invalid configuration."""
        # Invalid: 16 heads, 7 KV heads (16 % 7 != 0)
        with pytest.raises(ValueError, match="must be divisible"):
            Qwen3EncoderDecoderConfig(
                num_attention_heads=16,
                num_key_value_heads=7,
            )

    def test_validation_hidden_size_valid(self):
        """Test hidden_size validation with valid configuration."""
        config = Qwen3EncoderDecoderConfig(
            hidden_size=1024,
            num_attention_heads=16,
        )
        assert config.validate() is True

    def test_validation_hidden_size_invalid(self):
        """Test hidden_size validation with invalid configuration."""
        # Invalid: hidden_size not divisible by num_heads
        with pytest.raises(ValueError, match="must be divisible"):
            Qwen3EncoderDecoderConfig(
                hidden_size=1000,
                num_attention_heads=16,
            )

    def test_validation_sentinel_tokens_negative(self):
        """Test that negative sentinel tokens raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            Qwen3EncoderDecoderConfig(
                num_sentinel_tokens=-1,
                vocab_size=151935,
                sentinel_token_start_id=151936,
            )

    def test_validation_vocab_sentinel_mismatch(self):
        """Test that vocab_size must match sentinel configuration."""
        with pytest.raises(ValueError, match="must equal"):
            Qwen3EncoderDecoderConfig(
                vocab_size=160000,  # Wrong: should be 151936 + 100 = 152036
                num_sentinel_tokens=100,
                sentinel_token_start_id=151936,
            )


class TestSerialization:
    """Test suite for configuration serialization."""

    def test_save_and_load(self):
        """Test config serialization and deserialization."""
        config = Qwen3EncoderDecoderConfig(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1100,
            num_sentinel_tokens=100,
            sentinel_token_start_id=1000,
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
            assert loaded_config.num_attention_heads == 8
            assert loaded_config.num_key_value_heads == 4
            assert loaded_config.model_type == "qwen3_encdec"

    def test_to_json_string(self):
        """Test JSON string serialization."""
        config = Qwen3EncoderDecoderConfig()
        json_str = config.to_json_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "qwen3_encdec"
        assert parsed["hidden_size"] == 1024
        assert parsed["vocab_size"] == 152036

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Qwen3EncoderDecoderConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_type"] == "qwen3_encdec"
        assert config_dict["hidden_size"] == 1024


class TestFromQwen3Config:
    """Test suite for from_qwen3_config factory method."""

    def test_from_qwen3_config_mock(self):
        """Test creating config from mock Qwen3 config."""

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
        assert config.num_sentinel_tokens == 100

        # Should inherit other params
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 28
        assert config.num_attention_heads == 16

    def test_from_qwen3_config_custom_sentinels(self):
        """Test creating config with custom sentinel count."""

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

        config = Qwen3EncoderDecoderConfig.from_qwen3_config(
            mock_config, num_sentinel_tokens=50
        )

        assert config.vocab_size == 151936 + 50
        assert config.num_sentinel_tokens == 50

    def test_from_qwen3_config_with_overrides(self):
        """Test that kwargs override extracted values."""

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

        config = Qwen3EncoderDecoderConfig.from_qwen3_config(
            mock_config, num_hidden_layers=12
        )

        assert config.num_hidden_layers == 12

    def test_from_qwen3_config_missing_attrs(self):
        """Test that missing attributes raise ValueError."""

        class IncompleteConfig:
            hidden_size = 1024
            # Missing many required attributes

        with pytest.raises(ValueError, match="missing required attributes"):
            Qwen3EncoderDecoderConfig.from_qwen3_config(IncompleteConfig())

    def test_from_qwen3_config_with_rope_scaling(self):
        """Test that rope_scaling is properly transferred."""

        class MockQwen3ConfigWithRope:
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
            rope_scaling = {"type": "linear", "factor": 2.0}

        mock_config = MockQwen3ConfigWithRope()

        config = Qwen3EncoderDecoderConfig.from_qwen3_config(mock_config)

        assert config.rope_scaling == {"type": "linear", "factor": 2.0}


class TestRepr:
    """Test suite for __repr__ method."""

    def test_repr_basic(self):
        """Test __repr__ produces readable output."""
        config = Qwen3EncoderDecoderConfig()
        repr_str = repr(config)

        assert "Qwen3EncoderDecoderConfig" in repr_str
        assert "vocab_size=152036" in repr_str
        assert "hidden_size=1024" in repr_str
        assert "num_hidden_layers=28" in repr_str

    def test_repr_custom(self):
        """Test __repr__ with custom values."""
        config = Qwen3EncoderDecoderConfig(
            hidden_size=768,
            num_hidden_layers=12,
            vocab_size=32100,
            num_sentinel_tokens=100,
            sentinel_token_start_id=32000,
            num_attention_heads=12,
            num_key_value_heads=4,
        )
        repr_str = repr(config)

        assert "hidden_size=768" in repr_str
        assert "num_hidden_layers=12" in repr_str


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_zero_sentinel_tokens(self):
        """Test configuration with zero sentinel tokens."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=151936,
            num_sentinel_tokens=0,
            sentinel_token_start_id=151936,
        )

        assert config.num_sentinel_tokens == 0
        assert config.original_vocab_size == 151936

    def test_mha_config(self):
        """Test Multi-Head Attention (not GQA) configuration."""
        # When num_key_value_heads == num_attention_heads, it's MHA
        config = Qwen3EncoderDecoderConfig(
            num_attention_heads=16,
            num_key_value_heads=16,
        )

        assert config.num_key_value_groups == 1

    def test_extra_kwargs_preserved(self):
        """Test that extra kwargs are preserved in config."""
        config = Qwen3EncoderDecoderConfig(
            custom_param="test_value",
        )

        # Extra kwargs should be accessible
        assert hasattr(config, "custom_param")
        assert config.custom_param == "test_value"
