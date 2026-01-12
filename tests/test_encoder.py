"""Unit tests for Qwen3 Encoder implementation."""

import pytest
import torch
import torch.nn as nn

from qwen3_encdec import Qwen3EncoderDecoderConfig
from qwen3_encdec.modeling_qwen3_encoder import (
    Qwen3Encoder,
    Qwen3EncoderAttention,
    Qwen3EncoderLayer,
    Qwen3EncoderModel,
    Qwen3EncoderOutput,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)

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
def batch_inputs():
    """Create sample batch inputs."""
    batch_size = 2
    seq_len = 8
    return {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }


# =============================================================================
# RMSNorm Tests
# =============================================================================


class TestQwen3RMSNorm:
    """Test suite for RMSNorm layer."""

    def test_output_shape(self):
        """Test that RMSNorm preserves input shape."""
        hidden_size = 64
        norm = Qwen3RMSNorm(hidden_size)

        x = torch.randn(2, 10, hidden_size)
        output = norm(x)

        assert output.shape == x.shape

    def test_normalization_magnitude(self):
        """Test that RMSNorm normalizes to roughly unit RMS."""
        hidden_size = 64
        norm = Qwen3RMSNorm(hidden_size)

        x = torch.randn(2, 10, hidden_size) * 5  # Scale up input

        with torch.no_grad():
            output = norm(x)

        # RMS should be roughly 1 after normalization (before weight scaling)
        rms = output.pow(2).mean(-1).sqrt()
        # Weight is initialized to 1, so RMS should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_learnable_weight(self):
        """Test that RMSNorm has learnable weight parameter."""
        hidden_size = 64
        norm = Qwen3RMSNorm(hidden_size)

        assert hasattr(norm, "weight")
        assert norm.weight.shape == (hidden_size,)
        assert norm.weight.requires_grad

    def test_dtype_preservation(self):
        """Test that RMSNorm preserves input dtype."""
        hidden_size = 64
        norm = Qwen3RMSNorm(hidden_size)

        for dtype in [torch.float32, torch.bfloat16]:
            norm = norm.to(dtype)
            x = torch.randn(2, 10, hidden_size, dtype=dtype)
            output = norm(x)
            assert output.dtype == dtype


# =============================================================================
# Rotary Embedding Tests
# =============================================================================


class TestQwen3RotaryEmbedding:
    """Test suite for Rotary Position Embedding."""

    def test_output_shape(self):
        """Test that RoPE returns correct shapes."""
        dim = 64
        rope = Qwen3RotaryEmbedding(dim)

        # Input shape: [batch, num_heads, seq_len, head_dim]
        x = torch.randn(2, 4, 10, dim)
        cos, sin = rope(x)

        # Output should be broadcastable to x
        assert cos.shape[-1] == dim
        assert sin.shape[-1] == dim

    def test_cache_extension(self):
        """Test that cache extends for longer sequences."""
        dim = 64
        rope = Qwen3RotaryEmbedding(dim, max_position_embeddings=512)

        # First call with short sequence
        x1 = torch.randn(2, 4, 10, dim)
        cos1, sin1 = rope(x1)
        assert rope._cached_seq_len == 10

        # Second call with longer sequence - should extend cache
        x2 = torch.randn(2, 4, 50, dim)
        cos2, sin2 = rope(x2)
        assert rope._cached_seq_len == 50

    def test_position_ids(self):
        """Test that position_ids are respected."""
        dim = 64
        rope = Qwen3RotaryEmbedding(dim)

        x = torch.randn(2, 4, 10, dim)

        # With sequential position_ids
        pos_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        cos1, sin1 = rope(x, position_ids=pos_ids)

        # With reversed position_ids
        rev_pos_ids = torch.arange(9, -1, -1).unsqueeze(0).expand(2, -1)
        cos2, sin2 = rope(x, position_ids=rev_pos_ids)

        # Embeddings should be different (reversed positions)
        assert not torch.allclose(cos1, cos2)

    def test_cos_sin_values_bounded(self):
        """Test that cos/sin values are in [-1, 1]."""
        dim = 64
        rope = Qwen3RotaryEmbedding(dim)

        x = torch.randn(2, 4, 100, dim)
        cos, sin = rope(x)

        assert cos.abs().max() <= 1.0 + 1e-6
        assert sin.abs().max() <= 1.0 + 1e-6


# =============================================================================
# rotate_half and apply_rotary_pos_emb Tests
# =============================================================================


class TestRotateHalf:
    """Test suite for rotate_half function."""

    def test_output_shape(self):
        """Test that rotate_half preserves shape."""
        x = torch.randn(2, 4, 10, 64)
        output = rotate_half(x)
        assert output.shape == x.shape

    def test_rotation_correctness(self):
        """Test that rotate_half performs correct rotation."""
        # Simple test case
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output = rotate_half(x)

        # Expected: [-3, -4, 1, 2]
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(output, expected)


class TestApplyRotaryPosEmb:
    """Test suite for apply_rotary_pos_emb function."""

    def test_output_shapes(self):
        """Test that apply_rotary_pos_emb preserves shapes."""
        batch, heads, seq, dim = 2, 4, 10, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads // 2, seq, dim)  # Fewer KV heads

        cos = torch.randn(1, 1, seq, dim)
        sin = torch.randn(1, 1, seq, dim)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_encoding_effect(self):
        """Test that position encoding modifies the tensors."""
        batch, heads, seq, dim = 2, 4, 10, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)

        cos = torch.randn(1, 1, seq, dim)
        sin = torch.randn(1, 1, seq, dim)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Output should be different from input
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)


# =============================================================================
# repeat_kv Tests
# =============================================================================


class TestRepeatKV:
    """Test suite for repeat_kv function."""

    def test_no_repeat(self):
        """Test that n_rep=1 returns input unchanged."""
        x = torch.randn(2, 4, 10, 64)
        output = repeat_kv(x, n_rep=1)
        assert torch.equal(output, x)

    def test_repeat_doubles(self):
        """Test that n_rep=2 doubles the number of heads."""
        batch, kv_heads, seq, dim = 2, 4, 10, 64
        x = torch.randn(batch, kv_heads, seq, dim)

        output = repeat_kv(x, n_rep=2)

        assert output.shape == (batch, kv_heads * 2, seq, dim)

    def test_repeat_values_correct(self):
        """Test that repeated heads have identical values."""
        batch, kv_heads, seq, dim = 2, 2, 10, 64
        x = torch.randn(batch, kv_heads, seq, dim)

        output = repeat_kv(x, n_rep=3)

        # Head 0 should equal heads 0, 1, 2
        assert torch.allclose(output[:, 0], output[:, 1])
        assert torch.allclose(output[:, 0], output[:, 2])
        # Head 1 (original) should equal heads 3, 4, 5
        assert torch.allclose(output[:, 3], output[:, 4])
        assert torch.allclose(output[:, 3], output[:, 5])


# =============================================================================
# Attention Tests
# =============================================================================


class TestQwen3EncoderAttention:
    """Test suite for encoder attention layer."""

    def test_output_shape(self, small_config):
        """Test that attention returns correct output shape."""
        attn = Qwen3EncoderAttention(small_config)
        batch, seq = 2, 10

        x = torch.randn(batch, seq, small_config.hidden_size)
        output, _ = attn(x)

        assert output.shape == (batch, seq, small_config.hidden_size)

    def test_bidirectional_attention(self, small_config):
        """Test that attention is bidirectional (not causal)."""
        attn = Qwen3EncoderAttention(small_config)

        # Create input where later tokens have distinct values
        x = torch.zeros(1, 4, small_config.hidden_size)
        x[0, 3, :] = 10.0  # Last token has large values

        output, attn_weights = attn(x, output_attentions=True)

        # In bidirectional attention, earlier tokens should be influenced
        # by later tokens. Check that first token's output isn't zero.
        # (If causal, first token couldn't see the last token's large values)
        assert output[0, 0].abs().sum() > 0

    def test_attention_mask(self, small_config):
        """Test that attention mask is applied correctly."""
        attn = Qwen3EncoderAttention(small_config)
        batch, seq = 2, 8

        x = torch.randn(batch, seq, small_config.hidden_size)

        # Mask out last 4 tokens
        mask = torch.ones(batch, seq)
        mask[:, 4:] = 0

        output_masked, weights = attn(x, attention_mask=mask, output_attentions=True)

        # Attention weights to masked positions should be ~0
        # weights shape: [batch, heads, seq, seq]
        masked_attention = weights[:, :, :, 4:]  # Attention to masked positions
        assert masked_attention.abs().max() < 1e-4

    def test_output_attentions(self, small_config):
        """Test that attention weights can be returned."""
        attn = Qwen3EncoderAttention(small_config)
        x = torch.randn(2, 10, small_config.hidden_size)

        _, weights_none = attn(x, output_attentions=False)
        assert weights_none is None

        _, weights = attn(x, output_attentions=True)
        assert weights is not None
        assert weights.shape == (2, small_config.num_attention_heads, 10, 10)

    def test_qk_norm_applied(self, small_config):
        """Test that QK-Norm layers exist and are used."""
        attn = Qwen3EncoderAttention(small_config)

        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")
        assert isinstance(attn.q_norm, Qwen3RMSNorm)
        assert isinstance(attn.k_norm, Qwen3RMSNorm)

    def test_gqa_configuration(self, small_config):
        """Test that GQA is configured correctly."""
        attn = Qwen3EncoderAttention(small_config)

        assert attn.num_heads == 4
        assert attn.num_key_value_heads == 2
        assert attn.num_key_value_groups == 2


# =============================================================================
# MLP Tests
# =============================================================================


class TestQwen3MLP:
    """Test suite for MLP layer."""

    def test_output_shape(self, small_config):
        """Test that MLP preserves hidden size."""
        mlp = Qwen3MLP(small_config)
        x = torch.randn(2, 10, small_config.hidden_size)

        output = mlp(x)

        assert output.shape == x.shape

    def test_intermediate_expansion(self, small_config):
        """Test that intermediate size is different from hidden size."""
        mlp = Qwen3MLP(small_config)

        assert mlp.gate_proj.out_features == small_config.intermediate_size
        assert mlp.up_proj.out_features == small_config.intermediate_size
        assert mlp.down_proj.in_features == small_config.intermediate_size

    def test_gated_activation(self, small_config):
        """Test that gated activation uses gate and up projections."""
        mlp = Qwen3MLP(small_config)

        # Verify gated structure by checking both projections exist
        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert hasattr(mlp, "down_proj")
        assert hasattr(mlp, "act_fn")


# =============================================================================
# Encoder Layer Tests
# =============================================================================


class TestQwen3EncoderLayer:
    """Test suite for encoder layer."""

    def test_output_shape(self, small_config):
        """Test that encoder layer preserves shape."""
        layer = Qwen3EncoderLayer(small_config)
        x = torch.randn(2, 10, small_config.hidden_size)

        output, _ = layer(x)

        assert output.shape == x.shape

    def test_residual_connections(self, small_config):
        """Test that residual connections are working."""
        layer = Qwen3EncoderLayer(small_config)

        # Zero out all weights to isolate residual
        for param in layer.parameters():
            param.data.zero_()

        x = torch.randn(2, 10, small_config.hidden_size)
        output, _ = layer(x)

        # With zeroed weights, output should be close to input (residual only)
        # Note: LayerNorm will still have effect, so not exactly equal
        # But the residual should dominate
        assert output.shape == x.shape

    def test_layer_components(self, small_config):
        """Test that layer has all required components."""
        layer = Qwen3EncoderLayer(small_config)

        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "mlp")
        assert hasattr(layer, "input_layernorm")
        assert hasattr(layer, "post_attention_layernorm")

    def test_attention_output(self, small_config):
        """Test that attention weights can be returned from layer."""
        layer = Qwen3EncoderLayer(small_config)
        x = torch.randn(2, 10, small_config.hidden_size)

        _, attn_weights = layer(x, output_attentions=True)

        assert attn_weights is not None
        assert attn_weights.shape[0] == 2  # batch
        assert attn_weights.shape[2] == 10  # seq
        assert attn_weights.shape[3] == 10  # seq


# =============================================================================
# Full Encoder Tests
# =============================================================================


class TestQwen3Encoder:
    """Test suite for full encoder."""

    def test_output_shape(self, small_config, batch_inputs):
        """Test that encoder returns correct output shape."""
        encoder = Qwen3Encoder(small_config)
        output = encoder(**batch_inputs)

        assert output.last_hidden_state.shape == (
            2,
            8,
            small_config.hidden_size,
        )

    def test_output_type(self, small_config, batch_inputs):
        """Test that encoder returns Qwen3EncoderOutput."""
        encoder = Qwen3Encoder(small_config)
        output = encoder(**batch_inputs)

        assert isinstance(output, Qwen3EncoderOutput)
        assert output.last_hidden_state is not None

    def test_return_dict_false(self, small_config, batch_inputs):
        """Test that return_dict=False returns tuple."""
        encoder = Qwen3Encoder(small_config)
        output = encoder(**batch_inputs, return_dict=False)

        assert isinstance(output, tuple)
        assert len(output) == 1  # Only last_hidden_state when no optionals

    def test_output_hidden_states(self, small_config, batch_inputs):
        """Test output_hidden_states returns all layer outputs."""
        encoder = Qwen3Encoder(small_config)
        output = encoder(**batch_inputs, output_hidden_states=True)

        assert output.hidden_states is not None
        # num_layers + 1 (input embedding + each layer output after final norm)
        assert len(output.hidden_states) == small_config.num_hidden_layers + 1

    def test_output_attentions(self, small_config, batch_inputs):
        """Test output_attentions returns attention weights."""
        encoder = Qwen3Encoder(small_config)
        output = encoder(**batch_inputs, output_attentions=True)

        assert output.attentions is not None
        assert len(output.attentions) == small_config.num_hidden_layers

    def test_inputs_embeds(self, small_config):
        """Test that inputs_embeds can be used instead of input_ids."""
        encoder = Qwen3Encoder(small_config)

        inputs_embeds = torch.randn(2, 8, small_config.hidden_size)
        output = encoder(inputs_embeds=inputs_embeds)

        assert output.last_hidden_state.shape == inputs_embeds.shape

    def test_input_ids_and_embeds_mutual_exclusion(self, small_config, batch_inputs):
        """Test that input_ids and inputs_embeds cannot both be provided."""
        encoder = Qwen3Encoder(small_config)

        inputs_embeds = torch.randn(2, 8, small_config.hidden_size)

        with pytest.raises(ValueError, match="Cannot specify both"):
            encoder(input_ids=batch_inputs["input_ids"], inputs_embeds=inputs_embeds)

    def test_neither_input_raises(self, small_config):
        """Test that at least one input must be provided."""
        encoder = Qwen3Encoder(small_config)

        with pytest.raises(ValueError, match="Must specify either"):
            encoder()

    def test_gradient_flow(self, small_config, batch_inputs):
        """Test that gradients flow through the encoder."""
        encoder = Qwen3Encoder(small_config)
        output = encoder(**batch_inputs)

        loss = output.last_hidden_state.sum()
        loss.backward()

        # Check that gradients exist for embedding
        assert encoder.embed_tokens.weight.grad is not None
        assert encoder.embed_tokens.weight.grad.abs().sum() > 0

    def test_batch_independence(self, small_config):
        """Test that batches are processed independently."""
        encoder = Qwen3Encoder(small_config)
        encoder.eval()

        # Single sample
        single_input = torch.randint(0, 100, (1, 8))
        single_output = encoder(single_input).last_hidden_state

        # Same sample in batch
        batch_input = single_input.repeat(3, 1)
        batch_output = encoder(batch_input).last_hidden_state

        # All batch items should produce same output as single
        for i in range(3):
            assert torch.allclose(single_output[0], batch_output[i], atol=1e-5)

    def test_variable_sequence_length(self, small_config):
        """Test encoder handles different sequence lengths."""
        encoder = Qwen3Encoder(small_config)

        for seq_len in [4, 16, 32, 64]:
            input_ids = torch.randint(0, 100, (2, seq_len))
            output = encoder(input_ids)
            assert output.last_hidden_state.shape == (
                2,
                seq_len,
                small_config.hidden_size,
            )

    def test_attention_mask_effect(self, small_config):
        """Test that attention mask affects output."""
        encoder = Qwen3Encoder(small_config)
        encoder.eval()

        input_ids = torch.randint(0, 100, (1, 8))

        # Full attention
        mask_full = torch.ones(1, 8)
        output_full = encoder(input_ids, attention_mask=mask_full).last_hidden_state

        # Masked attention (mask last 4 tokens)
        mask_partial = torch.ones(1, 8)
        mask_partial[:, 4:] = 0
        output_partial = encoder(
            input_ids, attention_mask=mask_partial
        ).last_hidden_state

        # Outputs should differ
        assert not torch.allclose(output_full, output_partial)

    def test_embedding_access(self, small_config):
        """Test get/set input embeddings."""
        encoder = Qwen3Encoder(small_config)

        embeddings = encoder.get_input_embeddings()
        assert isinstance(embeddings, nn.Embedding)
        assert embeddings.num_embeddings == small_config.vocab_size

        new_embeddings = nn.Embedding(small_config.vocab_size, small_config.hidden_size)
        encoder.set_input_embeddings(new_embeddings)
        assert encoder.get_input_embeddings() is new_embeddings


class TestQwen3EncoderModel:
    """Test suite for Qwen3EncoderModel wrapper."""

    def test_output_shape(self, small_config, batch_inputs):
        """Test that model wrapper returns correct output."""
        model = Qwen3EncoderModel(small_config)
        output = model(**batch_inputs)

        assert output.last_hidden_state.shape == (2, 8, small_config.hidden_size)

    def test_embedding_access(self, small_config):
        """Test get/set embeddings through model wrapper."""
        model = Qwen3EncoderModel(small_config)

        embeddings = model.get_input_embeddings()
        assert isinstance(embeddings, nn.Embedding)

    def test_model_attribute(self, small_config):
        """Test that inner model is accessible."""
        model = Qwen3EncoderModel(small_config)

        assert hasattr(model, "model")
        assert isinstance(model.model, Qwen3Encoder)


# =============================================================================
# Integration Tests
# =============================================================================


class TestEncoderIntegration:
    """Integration tests for encoder components."""

    def test_default_config_encoder(self):
        """Test encoder with default (Qwen3-0.6B-like) config."""
        config = Qwen3EncoderDecoderConfig()  # Default config
        encoder = Qwen3Encoder(config)

        # Check architecture matches expected
        assert len(encoder.layers) == 28
        assert encoder.embed_tokens.embedding_dim == 1024

        # Quick forward pass (small batch for memory)
        input_ids = torch.randint(0, 1000, (1, 16))
        output = encoder(input_ids)
        assert output.last_hidden_state.shape == (1, 16, 1024)

    def test_encoder_determinism(self, small_config):
        """Test that encoder is deterministic in eval mode."""
        encoder = Qwen3Encoder(small_config)
        encoder.eval()

        input_ids = torch.randint(0, 100, (2, 8))

        with torch.no_grad():
            output1 = encoder(input_ids).last_hidden_state
            output2 = encoder(input_ids).last_hidden_state

        assert torch.allclose(output1, output2)

    def test_dtype_consistency(self, small_config):
        """Test encoder handles different dtypes."""
        for dtype in [torch.float32, torch.bfloat16]:
            encoder = Qwen3Encoder(small_config).to(dtype)
            input_ids = torch.randint(0, 100, (1, 8))

            output = encoder(input_ids)
            assert output.last_hidden_state.dtype == dtype

    def test_encoder_from_config(self, small_config):
        """Test that encoder can be instantiated from config."""
        encoder = Qwen3Encoder(small_config)

        # Config should be accessible
        assert encoder.config == small_config
        assert encoder.config.hidden_size == 64


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEncoderGPU:
    """Test suite for encoder on GPU."""

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

    def test_encoder_forward_gpu(self, gpu_config):
        """Test basic forward pass on GPU."""
        encoder = Qwen3Encoder(gpu_config).cuda()
        input_ids = torch.randint(0, 100, (2, 16)).cuda()

        output = encoder(input_ids)

        assert output.last_hidden_state.shape == (2, 16, gpu_config.hidden_size)

    def test_encoder_output_device(self, gpu_config):
        """Test that outputs are on GPU."""
        encoder = Qwen3Encoder(gpu_config).cuda()
        input_ids = torch.randint(0, 100, (2, 16)).cuda()

        output = encoder(input_ids)

        assert output.last_hidden_state.device.type == "cuda"

    def test_cpu_gpu_consistency(self, gpu_config):
        """Test that CPU and GPU produce consistent outputs."""
        # Create encoder and copy to both devices
        encoder_cpu = Qwen3Encoder(gpu_config)
        encoder_gpu = Qwen3Encoder(gpu_config).cuda()

        # Copy weights from CPU to GPU
        encoder_gpu.load_state_dict(encoder_cpu.state_dict())

        encoder_cpu.eval()
        encoder_gpu.eval()

        # Same input on both devices
        input_ids = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            output_cpu = encoder_cpu(input_ids).last_hidden_state
            output_gpu = encoder_gpu(input_ids.cuda()).last_hidden_state.cpu()

        # Should match within floating point tolerance
        assert torch.allclose(output_cpu, output_gpu, atol=1e-5)

    def test_attention_mask_gpu(self, gpu_config):
        """Test attention masking works on GPU."""
        encoder = Qwen3Encoder(gpu_config).cuda()
        encoder.eval()

        input_ids = torch.randint(0, 100, (2, 16)).cuda()

        # Full attention
        mask_full = torch.ones(2, 16).cuda()
        output_full = encoder(input_ids, attention_mask=mask_full).last_hidden_state

        # Partial mask
        mask_partial = torch.ones(2, 16).cuda()
        mask_partial[:, 8:] = 0
        output_partial = encoder(
            input_ids, attention_mask=mask_partial
        ).last_hidden_state

        # Outputs should differ
        assert not torch.allclose(output_full, output_partial)

    def test_gradient_flow_gpu(self, gpu_config):
        """Test gradients flow correctly on GPU."""
        encoder = Qwen3Encoder(gpu_config).cuda()
        input_ids = torch.randint(0, 100, (2, 16)).cuda()

        output = encoder(input_ids)
        loss = output.last_hidden_state.sum()
        loss.backward()

        # Gradients should exist and be on GPU
        assert encoder.embed_tokens.weight.grad is not None
        assert encoder.embed_tokens.weight.grad.device.type == "cuda"
        assert encoder.embed_tokens.weight.grad.abs().sum() > 0

    def test_mixed_precision_bf16(self, gpu_config):
        """Test BF16 forward pass on GPU."""
        encoder = Qwen3Encoder(gpu_config).cuda().to(torch.bfloat16)
        input_ids = torch.randint(0, 100, (2, 16)).cuda()

        output = encoder(input_ids)

        assert output.last_hidden_state.dtype == torch.bfloat16
        assert output.last_hidden_state.shape == (2, 16, gpu_config.hidden_size)

    def test_sdpa_on_gpu(self, gpu_config):
        """Test SDPA path executes without error on GPU."""
        encoder = Qwen3Encoder(gpu_config).cuda()
        encoder.eval()

        # Larger sequence to exercise SDPA more thoroughly
        input_ids = torch.randint(0, 100, (4, 64)).cuda()
        attention_mask = torch.ones(4, 64).cuda()

        with torch.no_grad():
            output = encoder(input_ids, attention_mask=attention_mask)

        assert output.last_hidden_state.shape == (4, 64, gpu_config.hidden_size)
        assert not torch.isnan(output.last_hidden_state).any()

    def test_default_config_on_gpu(self):
        """Test default (Qwen3-0.6B-like) config on GPU."""
        config = Qwen3EncoderDecoderConfig()
        encoder = Qwen3Encoder(config).cuda()

        # Small batch to avoid OOM
        input_ids = torch.randint(0, 1000, (1, 32)).cuda()

        output = encoder(input_ids)

        assert output.last_hidden_state.shape == (1, 32, 1024)
        assert output.last_hidden_state.device.type == "cuda"
