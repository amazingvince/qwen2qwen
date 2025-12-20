"""Tests for weight initialization utilities."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.qwen3_encdec import Qwen3EncoderDecoderConfig, Qwen3ForSeq2SeqLM
from src.qwen3_encdec.weight_initialization import (
    Qwen3WeightMapper,
    verify_gradient_flow,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_config():
    """Create small config for testing."""
    return Qwen3EncoderDecoderConfig(
        vocab_size=1100,  # 1000 + 100 sentinels
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        num_sentinel_tokens=100,
        sentinel_token_start_id=1000,
    )


@pytest.fixture
def small_model(small_config):
    """Create small model for testing."""
    return Qwen3ForSeq2SeqLM(small_config)


@pytest.fixture
def mock_qwen3_state_dict():
    """Create mock Qwen3 state dict with 2 layers."""
    hidden_size = 256
    intermediate_size = 512
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads

    state = {}

    # Embedding
    state["model.embed_tokens.weight"] = torch.randn(1000, hidden_size)

    # Two layers for testing
    for layer_idx in range(2):
        layer_prefix = f"model.layers.{layer_idx}"

        # Attention
        state[f"{layer_prefix}.self_attn.q_proj.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        state[f"{layer_prefix}.self_attn.k_proj.weight"] = torch.randn(
            num_kv_heads * head_dim, hidden_size
        )
        state[f"{layer_prefix}.self_attn.v_proj.weight"] = torch.randn(
            num_kv_heads * head_dim, hidden_size
        )
        state[f"{layer_prefix}.self_attn.o_proj.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        state[f"{layer_prefix}.self_attn.q_norm.weight"] = torch.randn(head_dim)
        state[f"{layer_prefix}.self_attn.k_norm.weight"] = torch.randn(head_dim)

        # MLP
        state[f"{layer_prefix}.mlp.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state[f"{layer_prefix}.mlp.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state[f"{layer_prefix}.mlp.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size
        )

        # Norms
        state[f"{layer_prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.randn(
            hidden_size
        )

    # Final norm
    state["model.norm.weight"] = torch.randn(hidden_size)

    return state


# =============================================================================
# TestQwen3WeightMapper
# =============================================================================


class TestQwen3WeightMapper:
    """Tests for Qwen3WeightMapper class."""

    def test_extend_embeddings_shape(self, mock_qwen3_state_dict):
        """Test embedding extension produces correct shape."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=100)

        original = mock_qwen3_state_dict["model.embed_tokens.weight"]
        original_size = original.shape[0]

        extended = mapper._extend_embeddings(original, 100)

        assert extended.shape == (original_size + 100, original.shape[1])

    def test_extend_embeddings_preserves_original(self, mock_qwen3_state_dict):
        """Test that original embeddings are preserved after extension."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=100)

        original = mock_qwen3_state_dict["model.embed_tokens.weight"]
        extended = mapper._extend_embeddings(original, 100)

        # Original embeddings should be unchanged
        assert torch.allclose(extended[: original.shape[0]], original)

    def test_extend_embeddings_sentinel_statistics(self, mock_qwen3_state_dict):
        """Test sentinel embeddings have similar statistics to original."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=1000)  # More samples

        original = mock_qwen3_state_dict["model.embed_tokens.weight"]
        extended = mapper._extend_embeddings(original, 1000)

        sentinel_part = extended[original.shape[0] :]

        # Statistics should be roughly similar (within reasonable tolerance)
        original_std = original.std()
        sentinel_std = sentinel_part.std()

        # Allow 30% deviation (random initialization has variance)
        assert abs(sentinel_std - original_std) < 0.3 * original_std

    def test_extend_embeddings_zero_sentinels(self, mock_qwen3_state_dict):
        """Test embedding extension with zero sentinels returns clone."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=0)

        original = mock_qwen3_state_dict["model.embed_tokens.weight"]
        extended = mapper._extend_embeddings(original, 0)

        # Should be unchanged
        assert extended.shape == original.shape
        assert torch.allclose(extended, original)

    def test_map_layer_weights_encoder(self, mock_qwen3_state_dict):
        """Test layer weight mapping for encoder."""
        mapper = Qwen3WeightMapper()

        mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")

        # Check encoder attention keys exist with correct prefix
        assert "model.encoder.layers.0.self_attn.q_proj.weight" in mapped
        assert "model.encoder.layers.0.self_attn.k_proj.weight" in mapped
        assert "model.encoder.layers.0.self_attn.v_proj.weight" in mapped
        assert "model.encoder.layers.0.self_attn.o_proj.weight" in mapped
        assert "model.encoder.layers.0.self_attn.q_norm.weight" in mapped
        assert "model.encoder.layers.0.self_attn.k_norm.weight" in mapped
        assert "model.encoder.layers.0.mlp.gate_proj.weight" in mapped
        assert "model.encoder.layers.0.input_layernorm.weight" in mapped
        assert "model.encoder.layers.0.post_attention_layernorm.weight" in mapped

    def test_map_layer_weights_decoder(self, mock_qwen3_state_dict):
        """Test layer weight mapping for decoder."""
        mapper = Qwen3WeightMapper()

        mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "decoder")

        # Check decoder uses self_attn prefix (not merged_attn)
        assert "model.decoder.layers.0.self_attn.q_proj.weight" in mapped
        assert "model.decoder.layers.0.self_attn.k_proj.weight" in mapped
        assert "model.decoder.layers.0.mlp.gate_proj.weight" in mapped

    def test_map_layer_weights_values_match(self, mock_qwen3_state_dict):
        """Test that mapped values match source values."""
        mapper = Qwen3WeightMapper()

        enc_mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")
        dec_mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "decoder")

        # Encoder and decoder should have same values (from same source)
        src_q = mock_qwen3_state_dict["model.layers.0.self_attn.q_proj.weight"]

        assert torch.allclose(
            enc_mapped["model.encoder.layers.0.self_attn.q_proj.weight"], src_q
        )
        assert torch.allclose(
            dec_mapped["model.decoder.layers.0.self_attn.q_proj.weight"], src_q
        )

    def test_map_layer_weights_layer_index(self, mock_qwen3_state_dict):
        """Test layer weight mapping respects layer index."""
        mapper = Qwen3WeightMapper()

        mapped_0 = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")
        mapped_1 = mapper._map_layer_weights(mock_qwen3_state_dict, 1, "encoder")

        # Different layer indices should map different weights
        assert "model.encoder.layers.0.self_attn.q_proj.weight" in mapped_0
        assert "model.encoder.layers.1.self_attn.q_proj.weight" in mapped_1

        # Values should be different (from different source layers)
        src_0 = mock_qwen3_state_dict["model.layers.0.self_attn.q_proj.weight"]
        src_1 = mock_qwen3_state_dict["model.layers.1.self_attn.q_proj.weight"]

        assert torch.allclose(
            mapped_0["model.encoder.layers.0.self_attn.q_proj.weight"], src_0
        )
        assert torch.allclose(
            mapped_1["model.encoder.layers.1.self_attn.q_proj.weight"], src_1
        )


# =============================================================================
# TestWeightInitializationIntegration
# =============================================================================


class TestWeightInitializationIntegration:
    """Integration tests for weight initialization."""

    def test_verify_gradient_flow_all_pass(self, small_model):
        """Test gradient flow verification with valid model."""
        results = verify_gradient_flow(
            small_model,
            batch_size=2,
            enc_seq_len=16,
            dec_seq_len=8,
        )

        # All checks should pass for a properly constructed model
        assert results["encoder_receives_gradients"]
        assert results["decoder_receives_gradients"]
        assert results["shared_embedding_gradients"]
        assert results["encoder_attention_gradients"]

    def test_tied_embeddings_after_initialization(self, small_model):
        """Test that embeddings are tied after model creation."""
        # Embeddings should be tied
        assert (
            small_model.model.shared.weight.data_ptr()
            == small_model.model.encoder.embed_tokens.weight.data_ptr()
        )
        assert (
            small_model.model.shared.weight.data_ptr()
            == small_model.model.decoder.embed_tokens.weight.data_ptr()
        )

    def test_map_weights_to_model(self, small_config, mock_qwen3_state_dict):
        """Test mapping mock weights to actual model."""
        # Create mock config
        mock_qwen3_config = MagicMock()
        mock_qwen3_config.num_hidden_layers = 2

        mapper = Qwen3WeightMapper(num_sentinel_tokens=100)
        mapper._qwen3_state_dict = mock_qwen3_state_dict
        mapper._qwen3_config = mock_qwen3_config

        # Create model
        model = Qwen3ForSeq2SeqLM(small_config)

        # Map weights
        num_mapped, num_total = mapper.map_weights(model)

        # Should have mapped some weights
        assert num_mapped > 0
        assert num_total > 0

        # Check encoder got the weights
        src_q = mock_qwen3_state_dict["model.layers.0.self_attn.q_proj.weight"]
        assert torch.allclose(
            model.model.encoder.layers[0].self_attn.q_proj.weight, src_q
        )

        # Check decoder got the weights
        assert torch.allclose(
            model.model.decoder.layers[0].self_attn.q_proj.weight, src_q
        )


# =============================================================================
# TestSaveLoad
# =============================================================================


class TestSaveLoad:
    """Tests for saving and loading initialized models."""

    def test_save_and_load_preserves_weights(self, small_model):
        """Test that save/load preserves all weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            small_model.save_pretrained(tmpdir)

            # Load
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)

            # Compare weights
            for name, param in small_model.named_parameters():
                loaded_param = dict(loaded.named_parameters())[name]
                assert torch.allclose(param, loaded_param), f"Mismatch in {name}"

    def test_save_and_load_preserves_tied_embeddings(self, small_model):
        """Test that tied embeddings remain tied after load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_model.save_pretrained(tmpdir)
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)

            # Check tying is preserved
            assert (
                loaded.model.shared.weight.data_ptr()
                == loaded.model.encoder.embed_tokens.weight.data_ptr()
            )


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_sentinel_tokens(self):
        """Test with no sentinel tokens."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=0)

        embeddings = torch.randn(1000, 256)
        extended = mapper._extend_embeddings(embeddings, 0)

        # Should be unchanged
        assert extended.shape == embeddings.shape
        assert torch.allclose(extended, embeddings)

    def test_large_sentinel_count(self):
        """Test with many sentinel tokens."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=1000)

        embeddings = torch.randn(100, 256)
        extended = mapper._extend_embeddings(embeddings, 1000)

        assert extended.shape == (1100, 256)

    def test_gradient_flow_on_gpu(self, small_model):
        """Test gradient flow verification on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        small_model = small_model.cuda()
        results = verify_gradient_flow(
            small_model,
            batch_size=2,
            enc_seq_len=16,
            dec_seq_len=8,
        )

        assert results["encoder_receives_gradients"]
        assert results["decoder_receives_gradients"]


# =============================================================================
# TestMapperConfiguration
# =============================================================================


class TestMapperConfiguration:
    """Tests for mapper configuration handling."""

    def test_create_config_from_mock_qwen3(self):
        """Test creating encoder-decoder config from mock Qwen3 config."""
        # Create a mock Qwen3 config
        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.num_hidden_layers = 28
        mock_config.num_attention_heads = 16
        mock_config.num_key_value_heads = 8
        mock_config.intermediate_size = 3072
        mock_config.vocab_size = 151936
        mock_config.max_position_embeddings = 32768
        mock_config.rms_norm_eps = 1e-6
        mock_config.rope_theta = 10000.0
        mock_config.head_dim = 64
        mock_config.attention_bias = False

        with patch(
            "src.qwen3_encdec.weight_initialization.AutoConfig.from_pretrained"
        ) as mock_auto:
            mock_auto.return_value = mock_config

            mapper = Qwen3WeightMapper(num_sentinel_tokens=100)
            config = mapper.create_encoder_decoder_config()

            assert config.hidden_size == 1024
            assert config.num_hidden_layers == 28
            assert config.vocab_size == 151936 + 100  # Extended
            assert config.num_sentinel_tokens == 100


# =============================================================================
# TestWeightMappingCompleteness
# =============================================================================


class TestWeightMappingCompleteness:
    """Tests to ensure all expected weights are mapped."""

    def test_all_attention_keys_mapped(self, mock_qwen3_state_dict):
        """Test all attention keys are mapped."""
        mapper = Qwen3WeightMapper()
        mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")

        expected_keys = [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "q_norm.weight",
            "k_norm.weight",
        ]

        for key in expected_keys:
            full_key = f"model.encoder.layers.0.self_attn.{key}"
            assert full_key in mapped, f"Missing key: {full_key}"

    def test_all_mlp_keys_mapped(self, mock_qwen3_state_dict):
        """Test all MLP keys are mapped."""
        mapper = Qwen3WeightMapper()
        mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")

        expected_keys = [
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        ]

        for key in expected_keys:
            full_key = f"model.encoder.layers.0.mlp.{key}"
            assert full_key in mapped, f"Missing key: {full_key}"

    def test_all_norm_keys_mapped(self, mock_qwen3_state_dict):
        """Test all norm keys are mapped."""
        mapper = Qwen3WeightMapper()
        mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")

        expected_keys = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]

        for key in expected_keys:
            full_key = f"model.encoder.layers.0.{key}"
            assert full_key in mapped, f"Missing key: {full_key}"
