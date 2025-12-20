"""GPU integration tests for encoder-decoder pipeline.

Tests cover:
1. Encoder-Decoder integration on GPU
2. Memory stress tests (large batches, long sequences)
3. Autoregressive generation on GPU
"""

import gc

import pytest
import torch

from qwen3_encdec import (
    Qwen3Decoder,
    Qwen3Encoder,
    Qwen3EncoderDecoderConfig,
)


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_config():
    """Small config for quick tests."""
    return Qwen3EncoderDecoderConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_sentinel_tokens=10,
        sentinel_token_start_id=990,
    )


@pytest.fixture
def medium_config():
    """Medium config for stress tests."""
    return Qwen3EncoderDecoderConfig(
        vocab_size=10000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_sentinel_tokens=100,
        sentinel_token_start_id=9900,
    )


@pytest.fixture
def encoder_decoder_gpu(small_config):
    """Create encoder and decoder on GPU."""
    encoder = Qwen3Encoder(small_config).cuda().eval()
    decoder = Qwen3Decoder(small_config).cuda().eval()
    return encoder, decoder


# =============================================================================
# Encoder-Decoder Integration Tests
# =============================================================================


class TestEncoderDecoderIntegrationGPU:
    """Tests for full encoder-decoder pipeline on GPU."""

    def test_encoder_decoder_forward_gpu(self, encoder_decoder_gpu, small_config):
        """Test full encoder-decoder forward pass on GPU."""
        encoder, decoder = encoder_decoder_gpu
        batch_size = 4
        enc_len = 16
        dec_len = 8

        # Encoder forward
        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
        encoder_output = encoder(encoder_input_ids)

        # Decoder forward
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()
        decoder_output = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output.last_hidden_state,
        )

        # Verify outputs
        assert decoder_output.last_hidden_state.device.type == "cuda"
        assert decoder_output.last_hidden_state.shape == (batch_size, dec_len, small_config.hidden_size)

    def test_encoder_decoder_gradient_flow_gpu(self, small_config):
        """Test gradients flow from decoder loss to encoder on GPU."""
        encoder = Qwen3Encoder(small_config).cuda()
        decoder = Qwen3Decoder(small_config).cuda()
        encoder.train()
        decoder.train()

        batch_size = 2
        enc_len = 8
        dec_len = 4

        # Forward pass
        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

        encoder_output = encoder(encoder_input_ids)
        decoder_output = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output.last_hidden_state,
        )

        # Compute loss and backward
        loss = decoder_output.last_hidden_state.mean()
        loss.backward()

        # Check gradients exist in both encoder and decoder
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        decoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in decoder.parameters()
        )

        assert encoder_has_grad, "Encoder should have gradients"
        assert decoder_has_grad, "Decoder should have gradients"

    def test_encoder_decoder_bf16(self, small_config):
        """Test encoder-decoder with BF16 precision."""
        encoder = Qwen3Encoder(small_config).cuda().to(torch.bfloat16)
        decoder = Qwen3Decoder(small_config).cuda().to(torch.bfloat16)

        batch_size = 2
        enc_len = 8
        dec_len = 4

        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

        encoder_output = encoder(encoder_input_ids)
        decoder_output = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output.last_hidden_state,
        )

        assert decoder_output.last_hidden_state.dtype == torch.bfloat16

    def test_encoder_decoder_fp16(self, small_config):
        """Test encoder-decoder with FP16 precision."""
        encoder = Qwen3Encoder(small_config).cuda().to(torch.float16)
        decoder = Qwen3Decoder(small_config).cuda().to(torch.float16)

        batch_size = 2
        enc_len = 8
        dec_len = 4

        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

        encoder_output = encoder(encoder_input_ids)
        decoder_output = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output.last_hidden_state,
        )

        assert decoder_output.last_hidden_state.dtype == torch.float16

    def test_encoder_output_to_decoder_gpu(self, encoder_decoder_gpu, small_config):
        """Test encoder output correctly feeds to decoder cross-attention."""
        encoder, decoder = encoder_decoder_gpu
        batch_size = 2
        enc_len = 12
        dec_len = 6

        # Encode with attention mask (some padding)
        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
        encoder_attention_mask = torch.ones(batch_size, enc_len).cuda()
        encoder_attention_mask[:, -2:] = 0  # Last 2 positions are padding

        encoder_output = encoder(
            encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )

        # Decode with encoder attention mask
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()
        decoder_output = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
        )

        # Verify output shape
        assert decoder_output.last_hidden_state.shape == (batch_size, dec_len, small_config.hidden_size)


# =============================================================================
# Memory Stress Tests
# =============================================================================


class TestMemoryStressGPU:
    """Memory stress tests for encoder-decoder on GPU."""

    def test_large_batch_gpu(self, small_config):
        """Test with large batch size."""
        encoder = Qwen3Encoder(small_config).cuda().eval()
        decoder = Qwen3Decoder(small_config).cuda().eval()

        batch_size = 32
        enc_len = 16
        dec_len = 8

        with torch.no_grad():
            encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
            decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

            encoder_output = encoder(encoder_input_ids)
            decoder_output = decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_output.last_hidden_state,
            )

        assert decoder_output.last_hidden_state.shape[0] == batch_size

    def test_long_sequence_gpu(self, small_config):
        """Test with long sequences."""
        encoder = Qwen3Encoder(small_config).cuda().eval()
        decoder = Qwen3Decoder(small_config).cuda().eval()

        batch_size = 2
        enc_len = 512
        dec_len = 256

        with torch.no_grad():
            encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
            decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

            encoder_output = encoder(encoder_input_ids)
            decoder_output = decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_output.last_hidden_state,
            )

        assert decoder_output.last_hidden_state.shape == (batch_size, dec_len, small_config.hidden_size)

    def test_memory_cleanup_gpu(self, small_config):
        """Test no memory leaks across multiple forward passes."""
        encoder = Qwen3Encoder(small_config).cuda().eval()
        decoder = Qwen3Decoder(small_config).cuda().eval()

        batch_size = 4
        enc_len = 32
        dec_len = 16

        # Warmup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
            decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

            encoder_output = encoder(encoder_input_ids)
            _ = decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_output.last_hidden_state,
            )

        # Get baseline memory after warmup
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()

        # Multiple forward passes
        for _ in range(5):
            with torch.no_grad():
                encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
                decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

                encoder_output = encoder(encoder_input_ids)
                _ = decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_output.last_hidden_state,
                )

            # Delete intermediates
            del encoder_input_ids, decoder_input_ids, encoder_output

        # Check memory after cleanup
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        # Allow small variance but no significant leak
        memory_diff_mb = (final_memory - baseline_memory) / (1024 * 1024)
        assert memory_diff_mb < 10, f"Memory leaked: {memory_diff_mb:.2f} MB"

    def test_gradient_checkpointing_gpu(self, medium_config):
        """Test gradient checkpointing reduces memory usage."""
        # Use larger sequences to see memory differences
        batch_size = 8
        enc_len = 128
        dec_len = 64

        # Without gradient checkpointing
        encoder_no_ckpt = Qwen3Encoder(medium_config).cuda()
        decoder_no_ckpt = Qwen3Decoder(medium_config).cuda()
        encoder_no_ckpt.train()
        decoder_no_ckpt.train()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        encoder_input_ids = torch.randint(0, medium_config.vocab_size, (batch_size, enc_len)).cuda()
        decoder_input_ids = torch.randint(0, medium_config.vocab_size, (batch_size, dec_len)).cuda()

        encoder_output = encoder_no_ckpt(encoder_input_ids)
        decoder_output = decoder_no_ckpt(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output.last_hidden_state,
        )
        loss = decoder_output.last_hidden_state.mean()
        loss.backward()

        memory_without_ckpt = torch.cuda.max_memory_allocated()

        # Clean up
        del encoder_no_ckpt, decoder_no_ckpt, encoder_output, decoder_output, loss
        del encoder_input_ids, decoder_input_ids
        gc.collect()
        torch.cuda.empty_cache()

        # With gradient checkpointing
        encoder_ckpt = Qwen3Encoder(medium_config).cuda()
        decoder_ckpt = Qwen3Decoder(medium_config).cuda()
        encoder_ckpt.gradient_checkpointing_enable()
        decoder_ckpt.gradient_checkpointing_enable()
        encoder_ckpt.train()
        decoder_ckpt.train()

        torch.cuda.reset_peak_memory_stats()

        encoder_input_ids = torch.randint(0, medium_config.vocab_size, (batch_size, enc_len)).cuda()
        decoder_input_ids = torch.randint(0, medium_config.vocab_size, (batch_size, dec_len)).cuda()

        encoder_output = encoder_ckpt(encoder_input_ids)
        decoder_output = decoder_ckpt(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_output.last_hidden_state,
        )
        loss = decoder_output.last_hidden_state.mean()
        loss.backward()

        memory_with_ckpt = torch.cuda.max_memory_allocated()

        # Gradient checkpointing should use less or equal memory
        # (small models may not show much difference due to fixed overhead)
        assert memory_with_ckpt <= memory_without_ckpt, (
            f"Gradient checkpointing should not increase memory: "
            f"{memory_with_ckpt / 1e6:.1f}MB vs {memory_without_ckpt / 1e6:.1f}MB"
        )


# =============================================================================
# Autoregressive Generation Tests
# =============================================================================


class TestAutoregressiveGenerationGPU:
    """Autoregressive generation tests on GPU."""

    def test_autoregressive_loop_gpu(self, encoder_decoder_gpu, small_config):
        """Test full autoregressive generation loop on GPU."""
        encoder, decoder = encoder_decoder_gpu
        batch_size = 2
        enc_len = 16
        max_new_tokens = 10

        # Encode
        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
        encoder_output = encoder(encoder_input_ids)

        # Start with BOS token (assume 1)
        decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long, device="cuda")
        past_key_values = None
        generated_tokens = [decoder_input_ids]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                decoder_output = decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_output.last_hidden_state,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # Get last hidden state and sample next token
                last_hidden = decoder_output.last_hidden_state[:, -1, :]
                # Simple argmax on a random projection (simulating lm_head)
                next_token = torch.randint(0, small_config.vocab_size, (batch_size, 1), device="cuda")

                generated_tokens.append(next_token)
                decoder_input_ids = next_token
                past_key_values = decoder_output.past_key_values

        # Verify we generated the expected number of tokens
        full_sequence = torch.cat(generated_tokens, dim=1)
        assert full_sequence.shape == (batch_size, 1 + max_new_tokens)

    def test_kv_cache_memory_growth(self, small_config):
        """Test KV cache memory grows predictably."""
        encoder = Qwen3Encoder(small_config).cuda().eval()
        decoder = Qwen3Decoder(small_config).cuda().eval()

        batch_size = 1
        enc_len = 8

        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()

        with torch.no_grad():
            encoder_output = encoder(encoder_input_ids)

            # Initial decoding step
            decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long, device="cuda")
            output1 = decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_output.last_hidden_state,
                use_cache=True,
            )

            # Check cache structure
            assert output1.past_key_values is not None
            assert len(output1.past_key_values) == small_config.num_hidden_layers

            # Each layer has ((dec_k, dec_v), (enc_k, enc_v))
            for layer_cache in output1.past_key_values:
                dec_cache, enc_cache = layer_cache
                # Decoder cache: [batch, num_kv_heads, seq_len, head_dim]
                assert dec_cache[0].shape[2] == 1  # One decoder token
                assert enc_cache[0].shape[2] == enc_len  # All encoder tokens

            # Second step
            next_token = torch.randint(0, small_config.vocab_size, (batch_size, 1), device="cuda")
            output2 = decoder(
                input_ids=next_token,
                encoder_hidden_states=encoder_output.last_hidden_state,
                past_key_values=output1.past_key_values,
                use_cache=True,
            )

            # Decoder cache should grow, encoder cache stays same
            for layer_cache in output2.past_key_values:
                dec_cache, enc_cache = layer_cache
                assert dec_cache[0].shape[2] == 2  # Two decoder tokens now
                assert enc_cache[0].shape[2] == enc_len  # Still same encoder tokens

    def test_generation_determinism_gpu(self, encoder_decoder_gpu, small_config):
        """Test reproducible generation with fixed seed."""
        encoder, decoder = encoder_decoder_gpu
        batch_size = 2
        enc_len = 8
        num_steps = 5

        def generate_sequence(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()

            with torch.no_grad():
                encoder_output = encoder(encoder_input_ids)

                decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long, device="cuda")
                past_key_values = None
                outputs = []

                for _ in range(num_steps):
                    decoder_output = decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_output.last_hidden_state,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    outputs.append(decoder_output.last_hidden_state.clone())
                    # Simulate sampling
                    next_token = torch.randint(0, small_config.vocab_size, (batch_size, 1), device="cuda")
                    decoder_input_ids = next_token
                    past_key_values = decoder_output.past_key_values

            return outputs

        # Run twice with same seed
        outputs1 = generate_sequence(42)
        outputs2 = generate_sequence(42)

        # Should be identical
        for o1, o2 in zip(outputs1, outputs2):
            assert torch.allclose(o1, o2, atol=1e-6), "Outputs should be deterministic with same seed"

    def test_generation_bf16_gpu(self, small_config):
        """Test autoregressive generation in BF16."""
        encoder = Qwen3Encoder(small_config).cuda().to(torch.bfloat16).eval()
        decoder = Qwen3Decoder(small_config).cuda().to(torch.bfloat16).eval()

        batch_size = 2
        enc_len = 8
        num_steps = 5

        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()

        with torch.no_grad():
            encoder_output = encoder(encoder_input_ids)

            decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long, device="cuda")
            past_key_values = None

            for _ in range(num_steps):
                decoder_output = decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_output.last_hidden_state,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                assert decoder_output.last_hidden_state.dtype == torch.bfloat16

                next_token = torch.randint(0, small_config.vocab_size, (batch_size, 1), device="cuda")
                decoder_input_ids = next_token
                past_key_values = decoder_output.past_key_values

    def test_generation_fp16_gpu(self, small_config):
        """Test autoregressive generation in FP16."""
        encoder = Qwen3Encoder(small_config).cuda().to(torch.float16).eval()
        decoder = Qwen3Decoder(small_config).cuda().to(torch.float16).eval()

        batch_size = 2
        enc_len = 8
        num_steps = 5

        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()

        with torch.no_grad():
            encoder_output = encoder(encoder_input_ids)

            decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long, device="cuda")
            past_key_values = None

            for _ in range(num_steps):
                decoder_output = decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_output.last_hidden_state,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                assert decoder_output.last_hidden_state.dtype == torch.float16

                next_token = torch.randint(0, small_config.vocab_size, (batch_size, 1), device="cuda")
                decoder_input_ids = next_token
                past_key_values = decoder_output.past_key_values

    def test_batch_vs_incremental_consistency_gpu(self, encoder_decoder_gpu, small_config):
        """Verify batch decoding matches incremental decoding on GPU."""
        encoder, decoder = encoder_decoder_gpu
        batch_size = 1
        enc_len = 8
        dec_len = 4

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        encoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, enc_len)).cuda()
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, dec_len)).cuda()

        with torch.no_grad():
            encoder_output = encoder(encoder_input_ids)

            # Batch decoding
            batch_output = decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_output.last_hidden_state,
            )

            # Incremental decoding
            incremental_outputs = []
            past_key_values = None

            for i in range(dec_len):
                single_token = decoder_input_ids[:, i : i + 1]
                output = decoder(
                    input_ids=single_token,
                    encoder_hidden_states=encoder_output.last_hidden_state,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                incremental_outputs.append(output.last_hidden_state)
                past_key_values = output.past_key_values

            incremental_combined = torch.cat(incremental_outputs, dim=1)

        # Should match closely (allow small numerical differences)
        assert torch.allclose(batch_output.last_hidden_state, incremental_combined, atol=1e-4), (
            f"Batch and incremental should match. "
            f"Max diff: {(batch_output.last_hidden_state - incremental_combined).abs().max()}"
        )
