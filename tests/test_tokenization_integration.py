"""Integration tests with real Qwen3 tokenizer.

These tests require network access to download the Qwen3 tokenizer
and are skipped if the tokenizer is not available.
"""

import pytest

# Try to import the tokenizer, skip all tests if it fails
try:
    from qwen3_encdec.tokenization_qwen3_encdec import (
        Qwen3EncoderDecoderTokenizer,
        apply_sentinel_corruption,
    )

    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


@pytest.mark.skipif(not TOKENIZER_AVAILABLE, reason="Tokenizer not available")
class TestRealTokenizerIntegration:
    """Integration tests with actual Qwen3 tokenizer."""

    @pytest.fixture(scope="class")
    def real_tokenizer(self):
        """Load real Qwen3 tokenizer (cached at class level)."""
        try:
            return Qwen3EncoderDecoderTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B", num_sentinel_tokens=100
            )
        except Exception as e:
            pytest.skip(f"Could not load Qwen3 tokenizer: {e}")

    def test_tokenizer_loads(self, real_tokenizer):
        """Test that tokenizer loads successfully."""
        assert real_tokenizer is not None
        assert real_tokenizer.num_sentinel_tokens == 100

    def test_vocab_size_includes_sentinels(self, real_tokenizer):
        """Test that vocab size includes sentinel tokens."""
        # Vocab size should be original + sentinels
        assert real_tokenizer.vocab_size == real_tokenizer.original_vocab_size + 100
        # Original vocab size should be less than total
        assert real_tokenizer.original_vocab_size < real_tokenizer.vocab_size

    def test_encode_regular_text(self, real_tokenizer):
        """Test encoding regular text without sentinels."""
        text = "Hello, world!"
        encoded = real_tokenizer.encode(text)

        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(isinstance(t, int) for t in encoded)

    def test_encode_with_sentinels(self, real_tokenizer):
        """Test encoding text containing sentinel tokens."""
        text = "The quick <extra_id_0> jumps over <extra_id_1> dog"
        encoded = real_tokenizer.encode(text)

        # Should contain sentinel token IDs
        sentinel_0_id = real_tokenizer.get_sentinel_token_id(0)
        sentinel_1_id = real_tokenizer.get_sentinel_token_id(1)

        assert sentinel_0_id in encoded
        assert sentinel_1_id in encoded

    def test_decode_with_sentinels(self, real_tokenizer):
        """Test decoding tokens containing sentinels."""
        # Encode some text first to get valid token IDs
        text = "Hello world"
        regular_tokens = real_tokenizer.encode(text)

        # Add a sentinel in the middle
        sentinel_0_id = real_tokenizer.get_sentinel_token_id(0)
        tokens = regular_tokens[:2] + [sentinel_0_id] + regular_tokens[2:]

        decoded = real_tokenizer.decode(tokens)
        assert "<extra_id_0>" in decoded

    def test_roundtrip_with_sentinels(self, real_tokenizer):
        """Test encode-decode roundtrip with sentinels."""
        original = "Hello <extra_id_0> world <extra_id_1> test"
        encoded = real_tokenizer.encode(original)
        decoded = real_tokenizer.decode(encoded)

        # Decoded should contain the sentinel tokens
        assert "<extra_id_0>" in decoded
        assert "<extra_id_1>" in decoded

    def test_apply_corruption_integration(self, real_tokenizer):
        """Test span corruption with real tokenizer."""
        text = "The quick brown fox jumps over the lazy dog"
        token_ids = real_tokenizer.encode(text, add_special_tokens=False)

        # Corrupt spans at positions 2-4 and 7-8
        spans_to_corrupt = [(2, 4), (7, 8)]
        enc_ids, dec_ids = apply_sentinel_corruption(
            token_ids, spans_to_corrupt, real_tokenizer
        )

        # Verify sentinels are in encoder output
        sentinel_0 = real_tokenizer.get_sentinel_token_id(0)
        sentinel_1 = real_tokenizer.get_sentinel_token_id(1)

        assert sentinel_0 in enc_ids
        assert sentinel_1 in enc_ids

        # Verify decoder starts with sentinels
        assert dec_ids[0] == sentinel_0

    def test_sentinel_tokens_are_special(self, real_tokenizer):
        """Test that sentinel tokens are registered as special tokens."""
        sentinel_token = real_tokenizer.get_sentinel_token(0)

        # The token should be in additional_special_tokens
        special_tokens = real_tokenizer.base_tokenizer.additional_special_tokens
        assert sentinel_token in special_tokens

    def test_pad_token_is_set(self, real_tokenizer):
        """Test that pad_token is set (should be eos_token if not originally set)."""
        assert real_tokenizer.pad_token_id is not None


@pytest.mark.skipif(not TOKENIZER_AVAILABLE, reason="Tokenizer not available")
class TestTokenizerSaveLoadIntegration:
    """Integration tests for save/load with real tokenizer."""

    @pytest.fixture(scope="class")
    def real_tokenizer(self):
        """Load real Qwen3 tokenizer."""
        try:
            return Qwen3EncoderDecoderTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B", num_sentinel_tokens=50
            )
        except Exception as e:
            pytest.skip(f"Could not load Qwen3 tokenizer: {e}")

    def test_save_and_reload(self, real_tokenizer, tmp_path):
        """Test saving and reloading tokenizer."""
        # Save
        save_dir = tmp_path / "tokenizer"
        real_tokenizer.save_pretrained(str(save_dir))

        # Reload
        loaded = Qwen3EncoderDecoderTokenizer.from_pretrained(str(save_dir))

        # Verify properties match
        assert loaded.num_sentinel_tokens == real_tokenizer.num_sentinel_tokens
        assert loaded.original_vocab_size == real_tokenizer.original_vocab_size
        assert loaded.vocab_size == real_tokenizer.vocab_size

        # Verify sentinel tokens work
        assert loaded.get_sentinel_token_id(0) == real_tokenizer.get_sentinel_token_id(0)

    def test_reload_encodes_correctly(self, real_tokenizer, tmp_path):
        """Test that reloaded tokenizer encodes same as original."""
        # Save
        save_dir = tmp_path / "tokenizer"
        real_tokenizer.save_pretrained(str(save_dir))

        # Reload
        loaded = Qwen3EncoderDecoderTokenizer.from_pretrained(str(save_dir))

        # Test encoding
        text = "Hello <extra_id_0> world"
        original_encoded = real_tokenizer.encode(text)
        loaded_encoded = loaded.encode(text)

        assert original_encoded == loaded_encoded
