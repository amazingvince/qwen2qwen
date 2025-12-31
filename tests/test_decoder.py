"""Unit tests for Qwen3 Decoder with merged attention."""

import pytest
import torch
import torch.nn as nn

from qwen3_encdec import Qwen3EncoderDecoderConfig
from qwen3_encdec.modeling_qwen3_decoder import (Qwen3Decoder,
                                                 Qwen3DecoderLayer,
                                                 Qwen3DecoderOutput,
                                                 Qwen3MergedAttention)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_config():
    """Create a small config for testing."""
    return Qwen3EncoderDecoderConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
        num_sentinel_tokens=10,
        sentinel_token_start_id=990,
    )


@pytest.fixture
def encoder_hidden():
    """Create sample encoder hidden states."""
    return torch.randn(2, 10, 64)


# =============================================================================
# Merged Attention Tests
# =============================================================================


class TestQwen3MergedAttention:
    """Test suite for merged attention."""

    def test_output_shape(self, small_config, encoder_hidden):
        """Test that merged attention returns correct output shape."""
        attn = Qwen3MergedAttention(small_config)
        decoder_hidden = torch.randn(2, 8, small_config.hidden_size)

        output, _, _ = attn(decoder_hidden, encoder_hidden)

        assert output.shape == decoder_hidden.shape

    def test_attention_weights_shape(self, small_config, encoder_hidden):
        """Test attention weights shape for merged attention."""
        attn = Qwen3MergedAttention(small_config)
        decoder_hidden = torch.randn(2, 8, small_config.hidden_size)

        _, attn_weights, _ = attn(
            decoder_hidden, encoder_hidden, output_attentions=True
        )

        # Shape: [batch, heads, dec_len, dec_len + enc_len]
        assert attn_weights.shape == (2, 4, 8, 18)

    def test_causal_masking_decoder_to_decoder(self, small_config):
        """Test that decoder-to-decoder attention is causal."""
        attn = Qwen3MergedAttention(small_config)
        attn.eval()

        decoder_hidden = torch.randn(1, 5, small_config.hidden_size)
        encoder_hidden = torch.randn(1, 3, small_config.hidden_size)

        with torch.no_grad():
            _, attn_weights, _ = attn(
                decoder_hidden, encoder_hidden, output_attentions=True
            )

        # Decoder portion is [:, :, :, :5]
        decoder_attn = attn_weights[:, :, :, :5]

        # Upper triangle should be ~0 (masked)
        for i in range(5):
            for j in range(i + 1, 5):
                assert decoder_attn[:, :, i, j].abs().max() < 1e-5

    def test_full_attention_decoder_to_encoder(self, small_config):
        """Test that decoder-to-encoder attention is full."""
        attn = Qwen3MergedAttention(small_config)
        attn.eval()

        decoder_hidden = torch.randn(1, 5, small_config.hidden_size)
        encoder_hidden = torch.randn(1, 3, small_config.hidden_size)

        with torch.no_grad():
            _, attn_weights, _ = attn(
                decoder_hidden, encoder_hidden, output_attentions=True
            )

        # Encoder portion is [:, :, :, 5:]
        encoder_attn = attn_weights[:, :, :, 5:]

        # All positions should have some attention (no masking)
        assert (encoder_attn > 0.01).any(dim=-1).all()

    def test_kv_cache_first_step(self, small_config, encoder_hidden):
        """Test KV cache creation on first step."""
        attn = Qwen3MergedAttention(small_config)

        decoder_hidden = torch.randn(2, 1, small_config.hidden_size)

        output, _, cache = attn(
            decoder_hidden, encoder_hidden, use_cache=True
        )

        assert cache is not None
        # Cache structure: (dec_k, dec_v, enc_k, enc_v) - flat format
        assert len(cache) == 4

        # Decoder cache should have 1 token
        assert cache[0].shape[2] == 1  # dec_k
        # Encoder cache should have 10 tokens
        assert cache[2].shape[2] == 10  # enc_k

    def test_kv_cache_incremental(self, small_config, encoder_hidden):
        """Test KV cache grows during incremental decoding."""
        attn = Qwen3MergedAttention(small_config)
        attn.eval()

        # First token
        decoder_hidden_1 = torch.randn(2, 1, small_config.hidden_size)
        with torch.no_grad():
            _, _, cache_1 = attn(
                decoder_hidden_1, encoder_hidden, use_cache=True
            )

        # Second token
        decoder_hidden_2 = torch.randn(2, 1, small_config.hidden_size)
        with torch.no_grad():
            _, _, cache_2 = attn(
                decoder_hidden_2, encoder_hidden,
                past_key_value=cache_1,
                use_cache=True,
            )

        # Decoder cache should grow (flat format: dec_k, dec_v, enc_k, enc_v)
        assert cache_2[0].shape[2] == 2  # dec_k grew to 2 tokens
        # Encoder cache should stay same
        assert cache_2[2].shape[2] == 10  # enc_k unchanged

    def test_encoder_padding_mask(self, small_config):
        """Test encoder padding mask is applied."""
        attn = Qwen3MergedAttention(small_config)
        attn.eval()

        decoder_hidden = torch.randn(2, 3, small_config.hidden_size)
        encoder_hidden = torch.randn(2, 5, small_config.hidden_size)

        # Mask last 2 positions in encoder
        encoder_attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ], dtype=torch.float32)

        with torch.no_grad():
            _, attn_weights, _ = attn(
                decoder_hidden, encoder_hidden,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=True,
            )

        # Encoder portion starts at position 3 (after decoder)
        encoder_attn = attn_weights[:, :, :, 3:]

        # Masked positions should have ~0 attention
        assert encoder_attn[0, :, :, 3:].max() < 1e-5  # Last 2 masked
        assert encoder_attn[1, :, :, 4:].max() < 1e-5  # Last 1 masked

    def test_qk_norm_applied(self, small_config):
        """Test that QK-Norm layers exist."""
        attn = Qwen3MergedAttention(small_config)

        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")

    def test_gqa_configuration(self, small_config):
        """Test that GQA is configured correctly."""
        attn = Qwen3MergedAttention(small_config)

        assert attn.num_heads == 4
        assert attn.num_key_value_heads == 2
        assert attn.num_key_value_groups == 2


# =============================================================================
# Decoder Layer Tests
# =============================================================================


class TestQwen3DecoderLayer:
    """Test suite for decoder layer."""

    def test_output_shape(self, small_config, encoder_hidden):
        """Test that decoder layer preserves shape."""
        layer = Qwen3DecoderLayer(small_config)
        decoder_hidden = torch.randn(2, 8, small_config.hidden_size)

        output, _, _ = layer(decoder_hidden, encoder_hidden)

        assert output.shape == decoder_hidden.shape

    def test_residual_connection(self, small_config, encoder_hidden):
        """Test residual connections work."""
        layer = Qwen3DecoderLayer(small_config)

        # Small weights to make residual dominant
        for p in layer.parameters():
            p.data.fill_(0.001)

        decoder_hidden = torch.randn(2, 8, small_config.hidden_size)
        output, _, _ = layer(decoder_hidden, encoder_hidden)

        # Output should be close to input when weights are small
        assert output.shape == decoder_hidden.shape

    def test_layer_components(self, small_config):
        """Test that layer has all required components."""
        layer = Qwen3DecoderLayer(small_config)

        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "mlp")
        assert hasattr(layer, "input_layernorm")
        assert hasattr(layer, "post_attention_layernorm")

    def test_kv_cache_passthrough(self, small_config, encoder_hidden):
        """Test that KV cache is passed through layer."""
        layer = Qwen3DecoderLayer(small_config)
        decoder_hidden = torch.randn(2, 1, small_config.hidden_size)

        _, _, cache = layer(
            decoder_hidden, encoder_hidden, use_cache=True
        )

        assert cache is not None


# =============================================================================
# Full Decoder Tests
# =============================================================================


class TestQwen3Decoder:
    """Test suite for full decoder."""

    def test_forward_basic(self, small_config, encoder_hidden):
        """Test basic forward pass."""
        decoder = Qwen3Decoder(small_config)
        input_ids = torch.randint(0, 100, (2, 8))

        output = decoder(input_ids, encoder_hidden_states=encoder_hidden)

        assert output.last_hidden_state.shape == (2, 8, small_config.hidden_size)

    def test_output_type(self, small_config, encoder_hidden):
        """Test that decoder returns Qwen3DecoderOutput."""
        decoder = Qwen3Decoder(small_config)
        input_ids = torch.randint(0, 100, (2, 8))

        output = decoder(input_ids, encoder_hidden_states=encoder_hidden)

        assert isinstance(output, Qwen3DecoderOutput)

    def test_forward_without_encoder_raises(self, small_config):
        """Test that missing encoder raises error."""
        decoder = Qwen3Decoder(small_config)
        input_ids = torch.randint(0, 100, (2, 8))

        with pytest.raises(ValueError, match="encoder_hidden_states"):
            decoder(input_ids)

    def test_output_hidden_states(self, small_config, encoder_hidden):
        """Test returning all hidden states."""
        decoder = Qwen3Decoder(small_config)
        input_ids = torch.randint(0, 100, (2, 8))

        output = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            output_hidden_states=True,
        )

        assert output.hidden_states is not None
        # num_layers + 1 (embedding + each layer output)
        assert len(output.hidden_states) == small_config.num_hidden_layers + 1

    def test_output_attentions(self, small_config, encoder_hidden):
        """Test returning attention weights."""
        decoder = Qwen3Decoder(small_config)
        input_ids = torch.randint(0, 100, (2, 8))

        output = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            output_attentions=True,
        )

        assert output.attentions is not None
        assert len(output.attentions) == small_config.num_hidden_layers

    def test_kv_cache_basic(self, small_config, encoder_hidden):
        """Test KV cache for generation."""
        decoder = Qwen3Decoder(small_config)

        # First token
        input_ids = torch.randint(0, 100, (2, 1))
        output_1 = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            use_cache=True,
        )

        assert output_1.past_key_values is not None
        assert len(output_1.past_key_values) == small_config.num_hidden_layers

        # Second token
        input_ids = torch.randint(0, 100, (2, 1))
        output_2 = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            past_key_values=output_1.past_key_values,
            use_cache=True,
        )

        assert output_2.past_key_values is not None

    def test_gradient_flow(self, small_config):
        """Test gradients flow through decoder."""
        decoder = Qwen3Decoder(small_config)
        input_ids = torch.randint(0, 100, (2, 8))
        encoder_hidden = torch.randn(2, 10, small_config.hidden_size, requires_grad=True)

        output = decoder(input_ids, encoder_hidden_states=encoder_hidden)
        loss = output.last_hidden_state.sum()
        loss.backward()

        # Gradients should flow to encoder
        assert encoder_hidden.grad is not None
        assert (encoder_hidden.grad != 0).any()

        # Gradients should exist for decoder params
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_inputs_embeds(self, small_config, encoder_hidden):
        """Test that inputs_embeds can be used instead of input_ids."""
        decoder = Qwen3Decoder(small_config)
        inputs_embeds = torch.randn(2, 8, small_config.hidden_size)

        output = decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden,
        )

        assert output.last_hidden_state.shape == inputs_embeds.shape

    def test_embedding_access(self, small_config):
        """Test get/set input embeddings."""
        decoder = Qwen3Decoder(small_config)

        embeddings = decoder.get_input_embeddings()
        assert isinstance(embeddings, nn.Embedding)
        assert embeddings.num_embeddings == small_config.vocab_size

        new_embeddings = nn.Embedding(small_config.vocab_size, small_config.hidden_size)
        decoder.set_input_embeddings(new_embeddings)
        assert decoder.get_input_embeddings() is new_embeddings


# =============================================================================
# Integration Tests
# =============================================================================


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
            num_sentinel_tokens=10,
            sentinel_token_start_id=90,
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

    def test_decoder_determinism(self, small_config, encoder_hidden):
        """Test that decoder is deterministic in eval mode."""
        decoder = Qwen3Decoder(small_config)
        decoder.eval()

        input_ids = torch.randint(0, 100, (2, 8))

        with torch.no_grad():
            output1 = decoder(input_ids, encoder_hidden_states=encoder_hidden).last_hidden_state
            output2 = decoder(input_ids, encoder_hidden_states=encoder_hidden).last_hidden_state

        assert torch.allclose(output1, output2)

    def test_dtype_consistency(self, small_config, encoder_hidden):
        """Test decoder handles different dtypes."""
        for dtype in [torch.float32, torch.bfloat16]:
            decoder = Qwen3Decoder(small_config).to(dtype)
            enc_hidden = encoder_hidden.to(dtype)
            input_ids = torch.randint(0, 100, (2, 8))

            output = decoder(input_ids, encoder_hidden_states=enc_hidden)
            assert output.last_hidden_state.dtype == dtype


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDecoderGPU:
    """Test suite for decoder on GPU."""

    @pytest.fixture
    def gpu_config(self):
        """Create a small config for GPU testing."""
        return Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
            num_sentinel_tokens=10,
            sentinel_token_start_id=990,
        )

    def test_decoder_forward_gpu(self, gpu_config):
        """Test basic forward pass on GPU."""
        decoder = Qwen3Decoder(gpu_config).cuda()
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        encoder_hidden = torch.randn(2, 10, gpu_config.hidden_size).cuda()

        output = decoder(input_ids, encoder_hidden_states=encoder_hidden)

        assert output.last_hidden_state.shape == (2, 8, gpu_config.hidden_size)

    def test_decoder_output_device(self, gpu_config):
        """Test that outputs are on GPU."""
        decoder = Qwen3Decoder(gpu_config).cuda()
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        encoder_hidden = torch.randn(2, 10, gpu_config.hidden_size).cuda()

        output = decoder(input_ids, encoder_hidden_states=encoder_hidden)

        assert output.last_hidden_state.device.type == "cuda"

    def test_cpu_gpu_consistency(self, gpu_config):
        """Test that CPU and GPU produce consistent outputs."""
        decoder_cpu = Qwen3Decoder(gpu_config)
        decoder_gpu = Qwen3Decoder(gpu_config).cuda()

        # Copy weights
        decoder_gpu.load_state_dict(decoder_cpu.state_dict())

        decoder_cpu.eval()
        decoder_gpu.eval()

        input_ids = torch.randint(0, 100, (2, 8))
        encoder_hidden = torch.randn(2, 10, gpu_config.hidden_size)

        with torch.no_grad():
            output_cpu = decoder_cpu(
                input_ids, encoder_hidden_states=encoder_hidden
            ).last_hidden_state
            output_gpu = decoder_gpu(
                input_ids.cuda(), encoder_hidden_states=encoder_hidden.cuda()
            ).last_hidden_state.cpu()

        assert torch.allclose(output_cpu, output_gpu, atol=1e-5)

    def test_kv_cache_gpu(self, gpu_config):
        """Test KV cache works on GPU."""
        decoder = Qwen3Decoder(gpu_config).cuda()
        encoder_hidden = torch.randn(2, 10, gpu_config.hidden_size).cuda()

        # First token
        input_ids = torch.randint(0, 100, (2, 1)).cuda()
        output_1 = decoder(
            input_ids, encoder_hidden_states=encoder_hidden, use_cache=True
        )

        # Second token
        input_ids = torch.randint(0, 100, (2, 1)).cuda()
        output_2 = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden,
            past_key_values=output_1.past_key_values,
            use_cache=True,
        )

        assert output_2.last_hidden_state.device.type == "cuda"

    def test_gradient_flow_gpu(self, gpu_config):
        """Test gradients flow correctly on GPU."""
        decoder = Qwen3Decoder(gpu_config).cuda()
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        # Create tensor directly on GPU to be a leaf tensor
        encoder_hidden = torch.randn(
            2, 10, gpu_config.hidden_size, device="cuda", requires_grad=True
        )

        output = decoder(input_ids, encoder_hidden_states=encoder_hidden)
        loss = output.last_hidden_state.sum()
        loss.backward()

        assert encoder_hidden.grad is not None
        assert encoder_hidden.grad.device.type == "cuda"

    def test_mixed_precision_bf16(self, gpu_config):
        """Test BF16 forward pass on GPU."""
        decoder = Qwen3Decoder(gpu_config).cuda().to(torch.bfloat16)
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        encoder_hidden = torch.randn(2, 10, gpu_config.hidden_size).cuda().to(torch.bfloat16)

        output = decoder(input_ids, encoder_hidden_states=encoder_hidden)

        assert output.last_hidden_state.dtype == torch.bfloat16

    def test_sdpa_on_gpu(self, gpu_config):
        """Test SDPA path executes without error on GPU."""
        decoder = Qwen3Decoder(gpu_config).cuda()
        decoder.eval()

        input_ids = torch.randint(0, 100, (4, 32)).cuda()
        encoder_hidden = torch.randn(4, 20, gpu_config.hidden_size).cuda()

        with torch.no_grad():
            output = decoder(input_ids, encoder_hidden_states=encoder_hidden)

        assert output.last_hidden_state.shape == (4, 32, gpu_config.hidden_size)
        assert not torch.isnan(output.last_hidden_state).any()
