"""Tests for encoder extraction utilities."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.qwen3_encdec import Qwen3EncoderDecoderConfig
from src.qwen3_encdec.encoder_only import (
    Qwen3EncoderConfig,
    Qwen3EncoderPooler,
    Qwen3EncoderPoolerOutput,
    Qwen3StandaloneEncoderModel,
)


# =============================================================================
# Test Qwen3EncoderConfig
# =============================================================================


class TestQwen3EncoderConfig:
    """Tests for encoder configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Qwen3EncoderConfig()

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 28
        assert config.pooling_mode == "mean"
        assert config.normalize_embeddings is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = Qwen3EncoderConfig(
            hidden_size=512,
            num_hidden_layers=12,
            pooling_mode="cls",
            normalize_embeddings=False,
        )

        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.pooling_mode == "cls"
        assert config.normalize_embeddings is False

    def test_from_encoder_decoder_config(self):
        """Test creating config from encoder-decoder config."""
        # Use default config which has valid sentinel token settings
        enc_dec_config = Qwen3EncoderDecoderConfig()

        encoder_config = Qwen3EncoderConfig.from_encoder_decoder_config(
            enc_dec_config,
            pooling_mode="cls",
            normalize_embeddings=False,
        )

        assert encoder_config.hidden_size == enc_dec_config.hidden_size
        assert encoder_config.num_hidden_layers == enc_dec_config.num_hidden_layers
        assert encoder_config.intermediate_size == enc_dec_config.intermediate_size
        assert encoder_config.pooling_mode == "cls"
        assert encoder_config.normalize_embeddings is False

    def test_config_model_type(self):
        """Test config model type is set correctly."""
        config = Qwen3EncoderConfig()
        assert config.model_type == "qwen3_encoder"


# =============================================================================
# Test Qwen3EncoderPooler
# =============================================================================


class TestQwen3EncoderPooler:
    """Tests for pooling strategies."""

    @pytest.fixture
    def hidden_states(self):
        """Sample hidden states (batch=2, seq=5, hidden=8)."""
        return torch.randn(2, 5, 8)

    @pytest.fixture
    def attention_mask(self):
        """Sample attention mask with padding."""
        return torch.tensor(
            [
                [1, 1, 1, 1, 1],  # No padding
                [1, 1, 1, 0, 0],  # 2 padding tokens
            ]
        )

    def test_mean_pooling(self, hidden_states, attention_mask):
        """Test mean pooling."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="mean",
            normalize_embeddings=False,
        )
        pooler = Qwen3EncoderPooler(config)

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (2, 8)

        # Verify first sequence (no padding)
        expected_0 = hidden_states[0].mean(dim=0)
        assert torch.allclose(output[0], expected_0)

        # Verify second sequence (with padding)
        expected_1 = hidden_states[1, :3].mean(dim=0)
        assert torch.allclose(output[1], expected_1)

    def test_cls_pooling(self, hidden_states, attention_mask):
        """Test CLS (first token) pooling."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="cls",
            normalize_embeddings=False,
        )
        pooler = Qwen3EncoderPooler(config)

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (2, 8)
        assert torch.allclose(output[0], hidden_states[0, 0])
        assert torch.allclose(output[1], hidden_states[1, 0])

    def test_last_pooling(self, hidden_states, attention_mask):
        """Test last token pooling."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="last",
            normalize_embeddings=False,
        )
        pooler = Qwen3EncoderPooler(config)

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (2, 8)
        # First sequence: last token is index 4
        assert torch.allclose(output[0], hidden_states[0, 4])
        # Second sequence: last non-padding is index 2
        assert torch.allclose(output[1], hidden_states[1, 2])

    def test_weighted_mean_pooling(self, hidden_states, attention_mask):
        """Test weighted mean pooling."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="weighted_mean",
            normalize_embeddings=False,
        )
        pooler = Qwen3EncoderPooler(config)

        output = pooler(hidden_states, attention_mask)

        assert output.shape == (2, 8)
        # Just verify output is reasonable (not NaN/Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_normalization(self, hidden_states, attention_mask):
        """Test L2 normalization."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="mean",
            normalize_embeddings=True,
        )
        pooler = Qwen3EncoderPooler(config)

        output = pooler(hidden_states, attention_mask)

        # Check L2 norm is 1
        norms = output.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_invalid_pooling_mode(self):
        """Test invalid pooling mode raises error."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="invalid",
            normalize_embeddings=False,
        )
        pooler = Qwen3EncoderPooler(config)

        hidden_states = torch.randn(2, 5, 8)
        attention_mask = torch.ones(2, 5)

        with pytest.raises(ValueError, match="Unknown pooling mode"):
            pooler(hidden_states, attention_mask)


# =============================================================================
# Test Qwen3StandaloneEncoderModel
# =============================================================================


class TestQwen3StandaloneEncoderModel:
    """Tests for standalone encoder model."""

    @pytest.fixture
    def small_config(self):
        """Small config for testing."""
        return Qwen3EncoderConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            pad_token_id=0,  # Valid pad token for small vocab
        )

    def test_forward_pass(self, small_config):
        """Test forward pass produces correct shapes."""
        model = Qwen3StandaloneEncoderModel(small_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(input_ids, attention_mask)

        assert isinstance(outputs, Qwen3EncoderPoolerOutput)
        assert outputs.last_hidden_state.shape == (
            batch_size,
            seq_len,
            small_config.hidden_size,
        )
        assert outputs.pooler_output.shape == (batch_size, small_config.hidden_size)

    def test_forward_without_attention_mask(self, small_config):
        """Test forward pass without explicit attention mask."""
        model = Qwen3StandaloneEncoderModel(small_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)

        assert outputs.last_hidden_state.shape == (
            batch_size,
            seq_len,
            small_config.hidden_size,
        )
        assert outputs.pooler_output.shape == (batch_size, small_config.hidden_size)

    def test_encode_method(self, small_config):
        """Test batch encoding method."""
        model = Qwen3StandaloneEncoderModel(small_config)

        # Create a simple tokenizer mock
        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                batch_size = len(texts) if isinstance(texts, list) else 1
                return {
                    "input_ids": torch.randint(
                        0, small_config.vocab_size, (batch_size, 10)
                    ),
                    "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
                }

        tokenizer = MockTokenizer()
        sentences = ["Hello world", "Test sentence"]

        embeddings = model.encode(sentences, tokenizer, show_progress=False)

        assert embeddings.shape == (2, small_config.hidden_size)

    def test_encode_single_sentence(self, small_config):
        """Test encoding single sentence."""
        model = Qwen3StandaloneEncoderModel(small_config)

        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                batch_size = len(texts) if isinstance(texts, list) else 1
                return {
                    "input_ids": torch.randint(
                        0, small_config.vocab_size, (batch_size, 10)
                    ),
                    "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
                }

        tokenizer = MockTokenizer()

        embeddings = model.encode("Single sentence", tokenizer, show_progress=False)

        assert embeddings.shape == (1, small_config.hidden_size)

    def test_output_hidden_states(self, small_config):
        """Test returning hidden states from all layers."""
        model = Qwen3StandaloneEncoderModel(small_config)

        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        outputs = model(input_ids, output_hidden_states=True)

        assert outputs.hidden_states is not None
        # num_layers + 1 for embeddings
        assert len(outputs.hidden_states) == small_config.num_hidden_layers + 1

    def test_pooling_mode_from_config(self, small_config):
        """Test pooling mode is read from config."""
        small_config.pooling_mode = "cls"
        model = Qwen3StandaloneEncoderModel(small_config)

        assert model.pooler.pooling_mode == "cls"

    def test_normalization_from_config(self, small_config):
        """Test normalization setting is read from config."""
        small_config.normalize_embeddings = False
        model = Qwen3StandaloneEncoderModel(small_config)

        assert model.pooler.normalize is False


# =============================================================================
# Test CheckpointAverager
# =============================================================================


class TestCheckpointAverager:
    """Tests for checkpoint averaging."""

    def test_find_checkpoints(self):
        """Test finding checkpoints by pattern."""
        from src.extraction.checkpoint_averaging import CheckpointAverager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock checkpoints
            for i in [100, 200, 500]:
                (tmpdir / f"checkpoint-{i}").mkdir()

            averager = CheckpointAverager(
                checkpoint_dir=str(tmpdir), output_path=str(tmpdir / "averaged")
            )

            checkpoints = averager.find_checkpoints()

            assert len(checkpoints) == 3
            assert checkpoints[0].name == "checkpoint-100"
            assert checkpoints[-1].name == "checkpoint-500"

    def test_average_checkpoints(self):
        """Test averaging checkpoint state dicts."""
        from src.extraction.checkpoint_averaging import CheckpointAverager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 mock checkpoints with known weights
            for i in range(3):
                cp_dir = tmpdir / f"checkpoint-{i * 1000}"
                cp_dir.mkdir()

                # Create mock state dict with known values
                state_dict = {
                    "layer.weight": torch.full((2, 2), float(i)),
                    "layer.bias": torch.full((2,), float(i)),
                }
                torch.save(state_dict, cp_dir / "pytorch_model.bin")

                # Create mock config
                with open(cp_dir / "config.json", "w") as f:
                    f.write("{}")

            # Average
            averager = CheckpointAverager(
                checkpoint_dir=str(tmpdir), output_path=str(tmpdir / "averaged")
            )

            checkpoints = averager.find_checkpoints()
            averaged_state = averager.average_checkpoints(checkpoints)

            # Average of 0, 1, 2 = 1
            assert torch.allclose(
                averaged_state["layer.weight"], torch.full((2, 2), 1.0)
            )
            assert torch.allclose(averaged_state["layer.bias"], torch.full((2,), 1.0))

    def test_average_last_n(self):
        """Test averaging last N checkpoints."""
        from src.extraction.checkpoint_averaging import CheckpointAverager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 5 mock checkpoints
            for i in range(5):
                cp_dir = tmpdir / f"checkpoint-{i * 1000}"
                cp_dir.mkdir()

                state_dict = {"layer.weight": torch.full((2, 2), float(i))}
                torch.save(state_dict, cp_dir / "pytorch_model.bin")

                with open(cp_dir / "config.json", "w") as f:
                    f.write("{}")

            # Average last 3
            averager = CheckpointAverager(
                checkpoint_dir=str(tmpdir), output_path=str(tmpdir / "averaged")
            )

            output_path = averager.average_last_n(n=3)

            assert output_path.exists()
            # Check for either format (safetensors or bin)
            model_file = output_path / "model.safetensors"
            if not model_file.exists():
                model_file = output_path / "pytorch_model.bin"
            assert model_file.exists()

            # Verify averaging (last 3 are indices 2, 3, 4 -> avg = 3)
            if model_file.suffix == ".safetensors":
                from safetensors.torch import load_file

                averaged = load_file(model_file)
            else:
                averaged = torch.load(model_file)
            assert torch.allclose(
                averaged["layer.weight"], torch.full((2, 2), 3.0)
            )


# =============================================================================
# Test Sentence Transformers Export
# =============================================================================


class TestSentenceTransformersExport:
    """Tests for sentence-transformers export."""

    def test_create_config(self):
        """Test creating sentence-transformers config."""
        from src.extraction.sentence_transformers_export import (
            create_sentence_transformers_config,
        )
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            create_sentence_transformers_config(
                output_path=tmpdir,
                hidden_size=1024,
                max_seq_length=512,
                pooling_mode="mean",
                normalize=True,
            )

            # Check modules.json
            with open(Path(tmpdir) / "modules.json") as f:
                modules = json.load(f)

            assert len(modules) == 3  # Transformer, Pooling, Normalize
            assert modules[0]["type"] == "sentence_transformers.models.Transformer"
            assert modules[1]["type"] == "sentence_transformers.models.Pooling"
            assert modules[2]["type"] == "sentence_transformers.models.Normalize"

            # Check pooling config
            with open(Path(tmpdir) / "1_Pooling" / "config.json") as f:
                pooling_config = json.load(f)

            assert pooling_config["word_embedding_dimension"] == 1024
            assert pooling_config["pooling_mode_mean_tokens"] is True
            assert pooling_config["pooling_mode_cls_token"] is False

    def test_create_config_without_normalize(self):
        """Test creating config without normalization."""
        from src.extraction.sentence_transformers_export import (
            create_sentence_transformers_config,
        )
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            create_sentence_transformers_config(
                output_path=tmpdir,
                hidden_size=768,
                max_seq_length=256,
                pooling_mode="cls",
                normalize=False,
            )

            with open(Path(tmpdir) / "modules.json") as f:
                modules = json.load(f)

            assert len(modules) == 2  # No Normalize
            assert not (Path(tmpdir) / "2_Normalize").exists()


# =============================================================================
# Integration Test: from_seq2seq
# =============================================================================


class TestFromSeq2Seq:
    """Tests for extracting encoder from seq2seq model."""

    @pytest.fixture
    def small_seq2seq_config(self):
        """Small config for testing."""
        # vocab_size = sentinel_token_start_id + num_sentinel_tokens
        # Use small values that satisfy the validation
        return Qwen3EncoderDecoderConfig(
            vocab_size=1100,  # 1000 + 100 sentinels
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            sentinel_token_start_id=1000,
            num_sentinel_tokens=100,
        )

    def test_from_seq2seq(self, small_seq2seq_config):
        """Test extracting encoder from seq2seq model."""
        from src.qwen3_encdec import Qwen3ForSeq2SeqLM

        # Create seq2seq model
        seq2seq = Qwen3ForSeq2SeqLM(small_seq2seq_config)

        # Extract encoder
        encoder = Qwen3StandaloneEncoderModel.from_seq2seq(
            seq2seq, pooling_mode="mean", normalize_embeddings=True
        )

        # Verify config
        assert encoder.config.hidden_size == 64
        assert encoder.config.num_hidden_layers == 2
        assert encoder.config.pooling_mode == "mean"

        # Verify forward pass works
        input_ids = torch.randint(0, small_seq2seq_config.vocab_size, (2, 10))
        outputs = encoder(input_ids)

        assert outputs.pooler_output.shape == (2, 64)

    def test_from_seq2seq_weights_match(self, small_seq2seq_config):
        """Test that extracted encoder produces same hidden states."""
        from src.qwen3_encdec import Qwen3ForSeq2SeqLM

        # Create seq2seq model
        seq2seq = Qwen3ForSeq2SeqLM(small_seq2seq_config)
        seq2seq.eval()

        # Extract encoder
        encoder = Qwen3StandaloneEncoderModel.from_seq2seq(seq2seq)
        encoder.eval()

        # Compare outputs
        input_ids = torch.randint(0, small_seq2seq_config.vocab_size, (2, 10))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            # Get encoder output from seq2seq
            seq2seq_output = seq2seq.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            # Get output from standalone encoder
            encoder_output = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        # Compare hidden states
        diff = (
            seq2seq_output.last_hidden_state - encoder_output.last_hidden_state
        ).abs()
        max_diff = diff.max().item()

        assert max_diff < 1e-5, f"Hidden states differ by {max_diff}"
