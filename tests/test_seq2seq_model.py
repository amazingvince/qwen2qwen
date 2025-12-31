"""Tests for Qwen3 Seq2Seq models."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from qwen3_encdec import (Qwen3Decoder, Qwen3Encoder,
                          Qwen3EncoderDecoderConfig, Qwen3EncoderModel,
                          Qwen3ForSeq2SeqLM, Qwen3Seq2SeqModel)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Small config for testing."""
    return Qwen3EncoderDecoderConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_sentinel_tokens=10,
        sentinel_token_start_id=990,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=0,
    )


@pytest.fixture
def seq2seq_model(config):
    """Create Qwen3Seq2SeqModel for testing."""
    return Qwen3Seq2SeqModel(config)


@pytest.fixture
def lm_model(config):
    """Create Qwen3ForSeq2SeqLM for testing."""
    return Qwen3ForSeq2SeqLM(config)


# =============================================================================
# TestQwen3Seq2SeqModel
# =============================================================================


class TestQwen3Seq2SeqModel:
    """Tests for base Qwen3Seq2SeqModel."""

    def test_init_from_config(self, config):
        """Test model creates encoder/decoder from config."""
        model = Qwen3Seq2SeqModel(config)

        assert model.shared is not None
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.shared.weight.shape == (config.vocab_size, config.hidden_size)

    def test_init_with_components(self, config):
        """Test model accepts pre-built encoder/decoder."""
        encoder = Qwen3Encoder(config)
        decoder = Qwen3Decoder(config)

        model = Qwen3Seq2SeqModel(config, encoder=encoder, decoder=decoder)

        assert model.encoder is encoder
        assert model.decoder is decoder

    def test_shared_embeddings(self, seq2seq_model):
        """Test encoder and decoder share the same embedding."""
        assert seq2seq_model.encoder.embed_tokens is seq2seq_model.shared
        assert seq2seq_model.decoder.embed_tokens is seq2seq_model.shared

    def test_forward_returns_hidden_states(self, seq2seq_model, config):
        """Test forward returns decoder hidden states (no logits)."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))
        decoder_input_ids = torch.randint(0, config.vocab_size - 10, (2, 6))

        outputs = seq2seq_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        )

        assert outputs.last_hidden_state.shape == (2, 6, config.hidden_size)
        assert outputs.encoder_last_hidden_state.shape == (2, 8, config.hidden_size)
        # No logits in base model
        assert not hasattr(outputs, "logits")

    def test_encoder_outputs_reuse(self, seq2seq_model, config):
        """Test encoder outputs can be reused."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))
        decoder_input_ids = torch.randint(0, config.vocab_size - 10, (2, 6))

        # Get encoder outputs
        encoder_outputs = seq2seq_model.encoder(input_ids)

        # Reuse encoder outputs
        outputs = seq2seq_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )

        assert outputs.last_hidden_state.shape == (2, 6, config.hidden_size)

    def test_output_hidden_states(self, seq2seq_model, config):
        """Test returning all hidden states."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))
        decoder_input_ids = torch.randint(0, config.vocab_size - 10, (2, 6))

        outputs = seq2seq_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
        )

        assert outputs.encoder_hidden_states is not None
        assert outputs.decoder_hidden_states is not None
        # Should have num_layers + 1 (for embeddings)
        assert len(outputs.encoder_hidden_states) == config.num_hidden_layers + 1

    def test_output_attentions(self, seq2seq_model, config):
        """Test returning attention weights."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))
        decoder_input_ids = torch.randint(0, config.vocab_size - 10, (2, 6))

        outputs = seq2seq_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
        )

        assert outputs.encoder_attentions is not None
        assert outputs.decoder_attentions is not None


# =============================================================================
# TestQwen3ForSeq2SeqLM
# =============================================================================


class TestQwen3ForSeq2SeqLM:
    """Tests for Qwen3ForSeq2SeqLM."""

    def test_initialization(self, lm_model, config):
        """Test model components exist."""
        assert lm_model.model is not None
        assert lm_model.lm_head is not None
        assert lm_model.lm_head.out_features == config.vocab_size

    def test_tied_embeddings(self, lm_model):
        """Test all embeddings are tied together."""
        # shared == encoder.embed_tokens == decoder.embed_tokens
        assert lm_model.model.shared is lm_model.model.encoder.embed_tokens
        assert lm_model.model.shared is lm_model.model.decoder.embed_tokens
        # lm_head.weight == shared.weight
        assert lm_model.lm_head.weight is lm_model.model.shared.weight

    def test_forward_basic(self, lm_model, config):
        """Test basic forward pass returns logits."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))
        decoder_input_ids = torch.randint(0, config.vocab_size - 10, (2, 6))

        outputs = lm_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        )

        assert outputs.logits.shape == (2, 6, config.vocab_size)
        assert outputs.loss is None  # No labels provided

    def test_forward_with_labels(self, lm_model, config):
        """Test forward pass with labels computes loss."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))
        labels = torch.randint(0, config.vocab_size - 10, (2, 6))

        outputs = lm_model(
            input_ids=input_ids,
            labels=labels,
        )

        assert outputs.loss is not None
        assert outputs.loss.ndim == 0  # Scalar
        assert outputs.logits.shape == (2, 6, config.vocab_size)

    def test_shift_right(self, lm_model, config):
        """Test label shifting for teacher forcing."""
        labels = torch.tensor([[10, 20, 30, 40]])

        shifted = lm_model._shift_right(labels)

        assert shifted[0, 0] == config.decoder_start_token_id
        assert shifted[0, 1] == 10
        assert shifted[0, 2] == 20
        assert shifted[0, 3] == 30

    def test_shift_right_replaces_ignore_index(self, lm_model, config):
        """Test that -100 is replaced with pad_token_id."""
        labels = torch.tensor([[10, -100, 30, 40]])

        shifted = lm_model._shift_right(labels)

        # -100 should be replaced with pad_token_id
        assert shifted[0, 2] == config.pad_token_id

    def test_encoder_outputs_reuse(self, lm_model, config):
        """Test encoder outputs can be cached for generation."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))

        # Get encoder outputs
        encoder_outputs = lm_model.model.encoder(input_ids)

        # Use cached encoder outputs
        decoder_input_ids = torch.randint(0, config.vocab_size - 10, (2, 6))
        outputs = lm_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )

        assert outputs.logits.shape == (2, 6, config.vocab_size)

    def test_gradient_flow(self, lm_model, config):
        """Test gradients flow through entire model."""
        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))
        labels = torch.randint(0, config.vocab_size - 10, (2, 6))

        outputs = lm_model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()

        # Check encoder has gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in lm_model.model.encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should have gradients"

        # Check decoder has gradients
        decoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in lm_model.model.decoder.parameters()
        )
        assert decoder_has_grad, "Decoder should have gradients"

        # Check shared embeddings have gradients
        assert lm_model.model.shared.weight.grad is not None

    def test_resize_embeddings(self, lm_model):
        """Test vocabulary resizing."""
        new_size = 1100
        lm_model.resize_token_embeddings(new_size)

        assert lm_model.model.shared.weight.shape[0] == new_size
        assert lm_model.lm_head.weight.shape[0] == new_size
        assert lm_model.config.vocab_size == new_size

        # Still tied
        assert lm_model.lm_head.weight is lm_model.model.shared.weight

    def test_kv_cache(self, lm_model, config):
        """Test KV cache for incremental decoding."""
        input_ids = torch.randint(0, config.vocab_size - 10, (1, 8))
        decoder_input_ids = torch.randint(0, config.vocab_size - 10, (1, 1))

        # First step - no cache
        outputs1 = lm_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            use_cache=True,
        )

        assert outputs1.past_key_values is not None

        # Second step - use cache
        next_token = torch.randint(0, config.vocab_size - 10, (1, 1))
        outputs2 = lm_model(
            encoder_outputs=(outputs1.encoder_last_hidden_state,),
            decoder_input_ids=next_token,
            past_key_values=outputs1.past_key_values,
            use_cache=True,
        )

        assert outputs2.past_key_values is not None
        # Cache should have grown (flat format: dec_k, dec_v, enc_k, enc_v)
        dec_cache_len = outputs2.past_key_values[0][0].shape[2]
        assert dec_cache_len == 2  # Two tokens now


# =============================================================================
# TestGeneration
# =============================================================================


class TestGeneration:
    """Tests for HuggingFace generation support."""

    @pytest.fixture
    def model(self, config):
        """Create model in eval mode."""
        model = Qwen3ForSeq2SeqLM(config)
        model.eval()
        return model

    def test_greedy_generation(self, model, config):
        """Test greedy decoding."""
        input_ids = torch.randint(2, config.vocab_size - 10, (1, 5))

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=False,
            )

        assert generated.shape[0] == 1
        assert generated.shape[1] <= 11  # max_new_tokens + decoder_start

    def test_sampling_generation(self, model, config):
        """Test sampling-based generation."""
        input_ids = torch.randint(2, config.vocab_size - 10, (1, 5))

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=True,
                temperature=1.0,
                top_k=50,
            )

        assert generated.shape[0] == 1

    def test_beam_search_generation(self, model, config):
        """Test beam search generation."""
        input_ids = torch.randint(2, config.vocab_size - 10, (1, 5))

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                num_beams=3,
                do_sample=False,
            )

        assert generated.shape[0] == 1

    def test_prepare_inputs_for_generation(self, model, config):
        """Test generation input preparation."""
        decoder_input_ids = torch.tensor([[0, 10, 20]])
        encoder_outputs = model.model.encoder(torch.randint(2, 100, (1, 5)))

        inputs = model.prepare_inputs_for_generation(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
        )

        assert "decoder_input_ids" in inputs
        assert "encoder_outputs" in inputs
        assert inputs["use_cache"] is True

    def test_prepare_inputs_with_cache(self, model, config):
        """Test that with past, only last token is used."""
        decoder_input_ids = torch.tensor([[0, 10, 20]])

        # Create mock past_key_values
        input_ids = torch.randint(2, config.vocab_size - 10, (1, 5))
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=torch.tensor([[0]]),
                use_cache=True,
            )

        inputs = model.prepare_inputs_for_generation(
            decoder_input_ids=decoder_input_ids,
            past_key_values=outputs.past_key_values,
            encoder_outputs=(outputs.encoder_last_hidden_state,),
        )

        # Only last token should be used
        assert inputs["decoder_input_ids"].shape == (1, 1)
        assert inputs["decoder_input_ids"][0, 0] == 20

    def test_reorder_cache(self, config):
        """Test cache reordering for beam search."""
        # Create mock cache structure: (dec_k, dec_v, enc_k, enc_v) - flat format
        batch_size = 4
        num_heads = config.num_key_value_heads
        seq_len = 3
        head_dim = config.head_dim

        past_key_values = []
        for _ in range(config.num_hidden_layers):
            dec_k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            dec_v = torch.randn(batch_size, num_heads, seq_len, head_dim)
            enc_k = torch.randn(batch_size, num_heads, 5, head_dim)
            enc_v = torch.randn(batch_size, num_heads, 5, head_dim)
            past_key_values.append((dec_k, dec_v, enc_k, enc_v))

        past_key_values = tuple(past_key_values)

        # Reorder: select beams [2, 0, 3, 1]
        beam_idx = torch.tensor([2, 0, 3, 1])
        reordered = Qwen3ForSeq2SeqLM._reorder_cache(past_key_values, beam_idx)

        # Check structure preserved (flat format: 4 tensors per layer)
        assert len(reordered) == config.num_hidden_layers
        for layer_cache in reordered:
            assert len(layer_cache) == 4  # dec_k, dec_v, enc_k, enc_v

        # Check reordering correct: first batch of reordered should be beam 2 from original
        assert torch.allclose(
            reordered[0][0][0], past_key_values[0][0][2]
        )  # First batch of dec_k from beam 2


# =============================================================================
# TestQwen3EncoderModel
# =============================================================================


class TestQwen3EncoderModelFromSeq2Seq:
    """Tests for encoder extraction."""

    def test_from_seq2seq(self, config):
        """Test extracting encoder from Seq2Seq model."""
        # Create and "train" a Seq2Seq model
        seq2seq = Qwen3ForSeq2SeqLM(config)

        # Modify a weight to verify transfer
        with torch.no_grad():
            seq2seq.model.encoder.layers[0].mlp.gate_proj.weight.fill_(0.5)

        # Extract encoder
        encoder_model = Qwen3EncoderModel.from_seq2seq(seq2seq)

        # Verify weights transferred
        assert torch.allclose(
            encoder_model.model.layers[0].mlp.gate_proj.weight,
            seq2seq.model.encoder.layers[0].mlp.gate_proj.weight,
        )

    def test_from_seq2seq_forward(self, config):
        """Test extracted encoder produces same outputs."""
        seq2seq = Qwen3ForSeq2SeqLM(config)
        encoder_model = Qwen3EncoderModel.from_seq2seq(seq2seq)

        input_ids = torch.randint(0, config.vocab_size - 10, (2, 8))

        with torch.no_grad():
            seq2seq_enc_out = seq2seq.model.encoder(input_ids)
            encoder_out = encoder_model(input_ids)

        assert torch.allclose(
            seq2seq_enc_out.last_hidden_state,
            encoder_out.last_hidden_state,
        )


# =============================================================================
# TestSaveLoad
# =============================================================================


class TestModelSaveLoad:
    """Test model serialization."""

    def test_save_and_load(self, config):
        """Test model save and load."""
        model = Qwen3ForSeq2SeqLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_pretrained(tmpdir)

            # Check files exist
            assert (Path(tmpdir) / "config.json").exists()
            assert (
                (Path(tmpdir) / "model.safetensors").exists()
                or (Path(tmpdir) / "pytorch_model.bin").exists()
            )

            # Load
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)

            # Verify config
            assert loaded.config.hidden_size == config.hidden_size

            # Verify weights (spot check)
            assert torch.allclose(
                model.model.shared.weight,
                loaded.model.shared.weight,
            )

    def test_tied_weights_after_load(self, config):
        """Test that weights remain tied after load."""
        model = Qwen3ForSeq2SeqLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)

            # Check weights are tied
            assert loaded.model.shared is loaded.model.encoder.embed_tokens
            assert loaded.model.shared is loaded.model.decoder.embed_tokens
            assert loaded.lm_head.weight is loaded.model.shared.weight


# =============================================================================
# TestGPU
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """GPU tests for Seq2Seq models."""

    @pytest.fixture
    def gpu_config(self):
        """Config for GPU tests."""
        return Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_sentinel_tokens=10,
            sentinel_token_start_id=990,
            decoder_start_token_id=0,
            pad_token_id=0,
        )

    def test_forward_gpu(self, gpu_config):
        """Test forward pass on GPU."""
        model = Qwen3ForSeq2SeqLM(gpu_config).cuda().eval()

        input_ids = torch.randint(0, gpu_config.vocab_size - 10, (2, 8)).cuda()
        labels = torch.randint(0, gpu_config.vocab_size - 10, (2, 6)).cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        assert outputs.logits.device.type == "cuda"
        assert outputs.loss.device.type == "cuda"

    def test_generation_gpu(self, gpu_config):
        """Test generation on GPU."""
        model = Qwen3ForSeq2SeqLM(gpu_config).cuda().eval()

        input_ids = torch.randint(2, gpu_config.vocab_size - 10, (1, 5)).cuda()

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=False,
            )

        assert generated.device.type == "cuda"

    def test_mixed_precision_bf16(self, gpu_config):
        """Test BF16 mixed precision."""
        model = Qwen3ForSeq2SeqLM(gpu_config).cuda().to(torch.bfloat16).eval()

        input_ids = torch.randint(0, gpu_config.vocab_size - 10, (2, 8)).cuda()
        labels = torch.randint(0, gpu_config.vocab_size - 10, (2, 6)).cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        assert outputs.logits.dtype == torch.bfloat16

    def test_gradient_checkpointing_gpu(self, gpu_config):
        """Test gradient checkpointing on GPU."""
        model = Qwen3ForSeq2SeqLM(gpu_config).cuda()
        model.gradient_checkpointing_enable()
        model.train()

        input_ids = torch.randint(0, gpu_config.vocab_size - 10, (2, 16)).cuda()
        labels = torch.randint(0, gpu_config.vocab_size - 10, (2, 8)).cuda()

        outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()

        # Check gradients exist
        assert model.model.shared.weight.grad is not None
