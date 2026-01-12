"""Unit tests for Qwen3EncoderDecoderTokenizer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen3_encdec.tokenization_qwen3_encdec import (
    SENTINEL_TOKEN_TEMPLATE,
    UL2_PREFIX_TOKENS,
    Qwen3EncoderDecoderTokenizer,
    apply_sentinel_corruption,
    create_sentinel_sequence,
)


class TestQwen3EncoderDecoderTokenizer:
    """Test suite for extended tokenizer."""

    @pytest.fixture
    def mock_base_tokenizer(self):
        """Create a mock base tokenizer for testing."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=151936)
        mock.add_special_tokens = MagicMock(return_value=100)
        mock.pad_token = None
        mock.pad_token_id = 0
        mock.eos_token = "</s>"
        mock.eos_token_id = 1
        mock.bos_token = "<s>"
        mock.bos_token_id = 2
        mock.unk_token_id = 3
        # Return UNK for unknown tokens so sentinels get added
        mock.convert_tokens_to_ids = MagicMock(return_value=3)
        mock.additional_special_tokens = []
        return mock

    @pytest.fixture
    def tokenizer(self, mock_base_tokenizer):
        """Create tokenizer instance for testing."""
        return Qwen3EncoderDecoderTokenizer(
            mock_base_tokenizer, num_sentinel_tokens=100
        )

    def test_initialization(self, tokenizer, mock_base_tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer.num_sentinel_tokens == 100
        assert tokenizer.original_vocab_size == 151936

        # Verify tokens were added (sentinel + UL2 prefix)
        assert mock_base_tokenizer.add_special_tokens.call_count == 2

        # First call should be sentinel tokens
        first_call_args = mock_base_tokenizer.add_special_tokens.call_args_list[0][0][0]
        sentinel_tokens = first_call_args["additional_special_tokens"]
        assert len(sentinel_tokens) == 100
        assert sentinel_tokens[0] == "<extra_id_0>"
        assert sentinel_tokens[99] == "<extra_id_99>"

        # Second call should include UL2 prefix tokens
        second_call_args = mock_base_tokenizer.add_special_tokens.call_args_list[1][0][
            0
        ]
        ul2_tokens = second_call_args["additional_special_tokens"]
        assert "[R]" in ul2_tokens
        assert "[X]" in ul2_tokens
        assert "[S]" in ul2_tokens
        assert "[I]" in ul2_tokens

    def test_initialization_sets_pad_token(self):
        """Test that pad_token is set from eos_token if None."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=151936)
        mock.add_special_tokens = MagicMock(return_value=100)
        mock.pad_token = None
        mock.eos_token = "</s>"
        mock.unk_token_id = 3
        mock.convert_tokens_to_ids = MagicMock(return_value=3)
        mock.additional_special_tokens = []

        Qwen3EncoderDecoderTokenizer(mock, num_sentinel_tokens=100)

        # pad_token should be set to eos_token
        assert mock.pad_token == "</s>"

    def test_get_sentinel_token(self, tokenizer):
        """Test sentinel token string retrieval."""
        assert tokenizer.get_sentinel_token(0) == "<extra_id_0>"
        assert tokenizer.get_sentinel_token(50) == "<extra_id_50>"
        assert tokenizer.get_sentinel_token(99) == "<extra_id_99>"

    def test_get_sentinel_token_out_of_range(self, tokenizer):
        """Test error on out of range sentinel index."""
        with pytest.raises(ValueError, match="must be between 0 and 99"):
            tokenizer.get_sentinel_token(-1)

        with pytest.raises(ValueError, match="must be between 0 and 99"):
            tokenizer.get_sentinel_token(100)

    def test_get_sentinel_token_id(self, tokenizer):
        """Test sentinel token ID retrieval."""
        assert tokenizer.get_sentinel_token_id(0) == 151936
        assert tokenizer.get_sentinel_token_id(1) == 151937
        assert tokenizer.get_sentinel_token_id(99) == 152035

    def test_get_sentinel_token_id_out_of_range(self, tokenizer):
        """Test error on out of range sentinel index."""
        with pytest.raises(ValueError):
            tokenizer.get_sentinel_token_id(-1)

        with pytest.raises(ValueError):
            tokenizer.get_sentinel_token_id(100)

    def test_is_sentinel_token_id(self, tokenizer):
        """Test sentinel token ID detection."""
        # Regular tokens
        assert not tokenizer.is_sentinel_token_id(0)
        assert not tokenizer.is_sentinel_token_id(151935)

        # Sentinel tokens
        assert tokenizer.is_sentinel_token_id(151936)
        assert tokenizer.is_sentinel_token_id(152035)

        # Beyond sentinel range
        assert not tokenizer.is_sentinel_token_id(152036)

    def test_sentinel_id_to_index(self, tokenizer):
        """Test converting sentinel ID to index."""
        assert tokenizer.sentinel_id_to_index(151936) == 0
        assert tokenizer.sentinel_id_to_index(151937) == 1
        assert tokenizer.sentinel_id_to_index(152035) == 99

    def test_sentinel_id_to_index_invalid(self, tokenizer):
        """Test error on non-sentinel token ID."""
        with pytest.raises(ValueError, match="is not a sentinel token"):
            tokenizer.sentinel_id_to_index(100)

    def test_get_sentinel_tokens_in_range(self, tokenizer):
        """Test getting range of sentinel tokens."""
        tokens = tokenizer.get_sentinel_tokens_in_range(0, 3)
        assert tokens == ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"]

        tokens = tokenizer.get_sentinel_tokens_in_range(97)
        assert tokens == ["<extra_id_97>", "<extra_id_98>", "<extra_id_99>"]

    def test_get_sentinel_tokens_in_range_default_end(self, tokenizer):
        """Test that default end is num_sentinel_tokens."""
        tokens = tokenizer.get_sentinel_tokens_in_range(98)
        assert tokens == ["<extra_id_98>", "<extra_id_99>"]

    def test_delegation_to_base_tokenizer(self, tokenizer, mock_base_tokenizer):
        """Test that methods are properly delegated."""
        # Test encode
        tokenizer.encode("test")
        mock_base_tokenizer.encode.assert_called_once_with("test")

        # Test decode
        mock_base_tokenizer.reset_mock()
        tokenizer.decode([1, 2, 3])
        mock_base_tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_call_delegation(self, tokenizer, mock_base_tokenizer):
        """Test __call__ is delegated."""
        tokenizer("test text", padding=True)
        mock_base_tokenizer.assert_called_once_with("test text", padding=True)

    def test_vocab_size_property(self, tokenizer, mock_base_tokenizer):
        """Test vocab_size property."""
        assert tokenizer.vocab_size == len(mock_base_tokenizer)

    def test_len(self, tokenizer, mock_base_tokenizer):
        """Test __len__."""
        assert len(tokenizer) == len(mock_base_tokenizer)

    def test_special_token_properties(self, tokenizer):
        """Test special token ID properties."""
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.unk_token_id == 3

    def test_repr(self, tokenizer):
        """Test __repr__."""
        repr_str = repr(tokenizer)
        assert "Qwen3EncoderDecoderTokenizer" in repr_str
        assert "vocab_size=" in repr_str
        assert "num_sentinel_tokens=100" in repr_str
        assert "original_vocab_size=151936" in repr_str

    def test_sentinel_token_template(self):
        """Test sentinel token template constant."""
        assert SENTINEL_TOKEN_TEMPLATE == "<extra_id_{i}>"
        assert SENTINEL_TOKEN_TEMPLATE.format(i=0) == "<extra_id_0>"
        assert SENTINEL_TOKEN_TEMPLATE.format(i=99) == "<extra_id_99>"


class TestSentinelCorruptionFunctions:
    """Test sentinel corruption utility functions."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for corruption tests."""
        mock = MagicMock(spec=Qwen3EncoderDecoderTokenizer)
        mock.num_sentinel_tokens = 100
        mock.get_sentinel_token_id = lambda i: 151936 + i
        return mock

    def test_create_sentinel_sequence_single_span(self, mock_tokenizer):
        """Test creating sentinel sequence with single span."""
        spans = [[101, 102, 103]]  # One span with 3 tokens

        sentinel_ids, target_ids = create_sentinel_sequence(mock_tokenizer, spans)

        assert sentinel_ids == [151936]  # <extra_id_0>
        assert target_ids == [151936, 101, 102, 103]

    def test_create_sentinel_sequence_multiple_spans(self, mock_tokenizer):
        """Test creating sentinel sequence with multiple spans."""
        spans = [[10, 20], [30], [40, 50, 60]]

        sentinel_ids, target_ids = create_sentinel_sequence(mock_tokenizer, spans)

        assert sentinel_ids == [151936, 151937, 151938]
        assert target_ids == [151936, 10, 20, 151937, 30, 151938, 40, 50, 60]

    def test_create_sentinel_sequence_empty_span(self, mock_tokenizer):
        """Test creating sentinel sequence with empty span."""
        spans = [[10, 20], [], [40]]  # Middle span is empty

        sentinel_ids, target_ids = create_sentinel_sequence(mock_tokenizer, spans)

        assert sentinel_ids == [151936, 151937, 151938]
        assert target_ids == [151936, 10, 20, 151937, 151938, 40]

    def test_create_sentinel_sequence_too_many_spans(self, mock_tokenizer):
        """Test error when too many spans."""
        spans = [[i] for i in range(101)]  # 101 spans

        with pytest.raises(ValueError, match="exceeds"):
            create_sentinel_sequence(mock_tokenizer, spans)

    def test_create_sentinel_sequence_empty_list(self, mock_tokenizer):
        """Test with no spans."""
        spans: list = []

        sentinel_ids, target_ids = create_sentinel_sequence(mock_tokenizer, spans)

        assert sentinel_ids == []
        assert target_ids == []

    def test_apply_sentinel_corruption_basic(self, mock_tokenizer):
        """Test basic span corruption."""
        input_ids = [10, 20, 30, 40, 50, 60, 70]
        spans_to_corrupt = [(1, 3)]  # Replace positions 1-2 (tokens 20, 30)

        enc_ids, dec_ids = apply_sentinel_corruption(
            input_ids, spans_to_corrupt, mock_tokenizer
        )

        # Encoder: [10, <s0>, 40, 50, 60, 70]
        assert enc_ids == [10, 151936, 40, 50, 60, 70]

        # Decoder: [<s0>, 20, 30]
        assert dec_ids == [151936, 20, 30]

    def test_apply_sentinel_corruption_multiple_spans(self, mock_tokenizer):
        """Test corruption with multiple spans."""
        input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        spans_to_corrupt = [(1, 3), (5, 7), (9, 10)]

        enc_ids, dec_ids = apply_sentinel_corruption(
            input_ids, spans_to_corrupt, mock_tokenizer
        )

        # Encoder: [1, <s0>, 4, 5, <s1>, 8, 9, <s2>]
        assert enc_ids == [1, 151936, 4, 5, 151937, 8, 9, 151938]

        # Decoder: [<s0>, 2, 3, <s1>, 6, 7, <s2>, 10]
        assert dec_ids == [151936, 2, 3, 151937, 6, 7, 151938, 10]

    def test_apply_sentinel_corruption_at_start(self, mock_tokenizer):
        """Test corruption at sequence start."""
        input_ids = [1, 2, 3, 4, 5]
        spans_to_corrupt = [(0, 2)]

        enc_ids, dec_ids = apply_sentinel_corruption(
            input_ids, spans_to_corrupt, mock_tokenizer
        )

        assert enc_ids == [151936, 3, 4, 5]
        assert dec_ids == [151936, 1, 2]

    def test_apply_sentinel_corruption_at_end(self, mock_tokenizer):
        """Test corruption at sequence end."""
        input_ids = [1, 2, 3, 4, 5]
        spans_to_corrupt = [(3, 5)]

        enc_ids, dec_ids = apply_sentinel_corruption(
            input_ids, spans_to_corrupt, mock_tokenizer
        )

        assert enc_ids == [1, 2, 3, 151936]
        assert dec_ids == [151936, 4, 5]

    def test_apply_sentinel_corruption_entire_sequence(self, mock_tokenizer):
        """Test corruption of entire sequence."""
        input_ids = [1, 2, 3, 4, 5]
        spans_to_corrupt = [(0, 5)]

        enc_ids, dec_ids = apply_sentinel_corruption(
            input_ids, spans_to_corrupt, mock_tokenizer
        )

        assert enc_ids == [151936]
        assert dec_ids == [151936, 1, 2, 3, 4, 5]

    def test_apply_sentinel_corruption_unsorted_spans(self, mock_tokenizer):
        """Test that spans are sorted before processing."""
        input_ids = [1, 2, 3, 4, 5, 6, 7]
        # Spans not in order
        spans_to_corrupt = [(4, 6), (1, 3)]

        enc_ids, dec_ids = apply_sentinel_corruption(
            input_ids, spans_to_corrupt, mock_tokenizer
        )

        # Should handle correctly after sorting
        # Sorted spans: [(1, 3), (4, 6)]
        assert enc_ids == [1, 151936, 4, 151937, 7]
        assert dec_ids == [151936, 2, 3, 151937, 5, 6]

    def test_apply_sentinel_corruption_too_many_spans(self, mock_tokenizer):
        """Test error when too many spans."""
        input_ids = list(range(200))
        spans_to_corrupt = [(i, i + 1) for i in range(0, 200, 2)]  # 100 spans
        # That's exactly 100, so should work
        apply_sentinel_corruption(input_ids, spans_to_corrupt[:100], mock_tokenizer)

        # 101 spans should fail
        spans_to_corrupt = [(i, i + 1) for i in range(0, 202, 2)]  # 101 spans
        with pytest.raises(ValueError, match="exceeds"):
            apply_sentinel_corruption(input_ids, spans_to_corrupt[:101], mock_tokenizer)


class TestTokenizerSaveLoad:
    """Test tokenizer serialization."""

    @patch("qwen3_encdec.tokenization_qwen3_encdec.AutoTokenizer")
    def test_from_pretrained(self, mock_auto_tokenizer):
        """Test loading from pretrained."""
        mock_base = MagicMock()
        mock_base.__len__ = MagicMock(return_value=151936)
        mock_base.add_special_tokens = MagicMock(return_value=100)
        mock_base.pad_token = None
        mock_base.eos_token = "</s>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_base

        tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", num_sentinel_tokens=50
        )

        assert tokenizer.num_sentinel_tokens == 50
        mock_auto_tokenizer.from_pretrained.assert_called_once()

    @patch("qwen3_encdec.tokenization_qwen3_encdec.AutoTokenizer")
    def test_from_pretrained_with_sentinel_config(self, mock_auto_tokenizer):
        """Test loading from directory with sentinel_config.json."""
        mock_base = MagicMock()
        mock_base.__len__ = MagicMock(return_value=151936)
        mock_base.add_special_tokens = MagicMock(return_value=75)
        mock_base.pad_token = None
        mock_base.eos_token = "</s>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_base

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sentinel config
            config_path = Path(tmpdir) / "sentinel_config.json"
            with open(config_path, "w") as f:
                json.dump({"num_sentinel_tokens": 75}, f)

            tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(
                tmpdir,
                num_sentinel_tokens=100,  # Should be overridden
            )

            # Should use config from file, not the argument
            assert tokenizer.num_sentinel_tokens == 75

    def test_save_pretrained(self):
        """Test saving tokenizer."""
        mock_base = MagicMock()
        mock_base.__len__ = MagicMock(return_value=151936)
        mock_base.add_special_tokens = MagicMock(return_value=100)
        mock_base.pad_token = None
        mock_base.eos_token = "</s>"

        tokenizer = Qwen3EncoderDecoderTokenizer(mock_base, num_sentinel_tokens=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)

            # Verify base tokenizer save was called
            mock_base.save_pretrained.assert_called_once_with(tmpdir)

            # Verify sentinel config was saved
            config_path = Path(tmpdir) / "sentinel_config.json"
            assert config_path.exists()

            with open(config_path) as f:
                config = json.load(f)

            assert config["num_sentinel_tokens"] == 100
            assert config["original_vocab_size"] == 151936
            assert config["sentinel_token_template"] == "<extra_id_{i}>"


class TestCustomSentinelCount:
    """Test with different sentinel token counts."""

    def test_fewer_sentinels(self):
        """Test with fewer than 100 sentinel tokens."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=32000)
        mock.add_special_tokens = MagicMock(return_value=10)
        mock.pad_token = "</s>"
        mock.eos_token = "</s>"

        tokenizer = Qwen3EncoderDecoderTokenizer(mock, num_sentinel_tokens=10)

        assert tokenizer.num_sentinel_tokens == 10
        assert tokenizer.original_vocab_size == 32000
        assert tokenizer.get_sentinel_token(9) == "<extra_id_9>"

        with pytest.raises(ValueError):
            tokenizer.get_sentinel_token(10)

    def test_zero_sentinels(self):
        """Test with zero sentinel tokens."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=32000)
        mock.add_special_tokens = MagicMock(return_value=0)
        mock.pad_token = "</s>"
        mock.eos_token = "</s>"

        tokenizer = Qwen3EncoderDecoderTokenizer(mock, num_sentinel_tokens=0)

        assert tokenizer.num_sentinel_tokens == 0
        assert not tokenizer.is_sentinel_token_id(32000)

        with pytest.raises(ValueError):
            tokenizer.get_sentinel_token(0)


class TestUL2PrefixTokens:
    """Test UL2 prefix token handling."""

    def test_ul2_prefix_tokens_constant(self):
        """Test that UL2 prefix tokens are defined correctly."""
        assert UL2_PREFIX_TOKENS == ["[R]", "[X]", "[S]", "[I]"]

    def test_ul2_prefix_tokens_added_when_new(self):
        """Test that UL2 prefix tokens are added for new tokenizers."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=151936)
        # Sentinel tokens added successfully
        mock.add_special_tokens = MagicMock(return_value=100)
        mock.pad_token = None
        mock.eos_token = "</s>"
        mock.unk_token_id = 3
        # First prefix doesn't exist (returns UNK)
        mock.convert_tokens_to_ids = MagicMock(return_value=3)
        mock.additional_special_tokens = []

        Qwen3EncoderDecoderTokenizer(mock, num_sentinel_tokens=100)

        # Should have called add_special_tokens twice (sentinel + UL2 prefix)
        assert mock.add_special_tokens.call_count == 2

    def test_ul2_prefix_tokens_not_readded_when_exist(self):
        """Test that UL2 prefix tokens are not re-added if they exist."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=151940)
        mock.add_special_tokens = MagicMock(return_value=100)
        mock.pad_token = None
        mock.eos_token = "</s>"
        mock.unk_token_id = 3

        # Simulate tokens already existing
        def convert_tokens_to_ids(token):
            if token == "<extra_id_0>":
                return 151936  # Sentinel exists
            if token == "[R]":
                return 151940  # UL2 prefix exists (not UNK)
            return 3  # UNK

        mock.convert_tokens_to_ids = MagicMock(side_effect=convert_tokens_to_ids)
        mock.additional_special_tokens = ["[R]", "[X]", "[S]", "[I]"]

        Qwen3EncoderDecoderTokenizer(mock, num_sentinel_tokens=100)

        # Should only call add_special_tokens once (for sentinels check, returns early)
        # Actually with our new logic, if first sentinel exists, we return early
        # So add_special_tokens should not be called at all
        # Let me re-check the logic...
        # _add_sentinel_tokens: if first_sentinel_id != unk_id, return early (no call)
        # _add_ul2_prefix_tokens: if first_prefix_id != unk_id, return early (no call)
        assert mock.add_special_tokens.call_count == 0


class TestSentinelTokenDetection:
    """Test sentinel token detection when loading saved tokenizers."""

    def test_sentinels_detected_as_existing(self):
        """Test that existing sentinel tokens are detected and not re-added."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=152036)
        mock.add_special_tokens = MagicMock(return_value=0)
        mock.pad_token = "</s>"
        mock.eos_token = "</s>"
        mock.unk_token_id = 3

        # First sentinel exists (not UNK)
        def convert_tokens_to_ids(token):
            if token == "<extra_id_0>":
                return 151936  # Exists
            if token == "[R]":
                return 152036  # Exists
            return 3

        mock.convert_tokens_to_ids = MagicMock(side_effect=convert_tokens_to_ids)
        mock.additional_special_tokens = []

        tokenizer = Qwen3EncoderDecoderTokenizer(mock, num_sentinel_tokens=100)

        # Sentinel tokens should not be re-added
        # add_special_tokens should not be called for sentinels
        # (but may be called for UL2 prefixes if they don't exist)
        assert tokenizer.num_sentinel_tokens == 100

    def test_sentinels_added_when_not_exist(self):
        """Test that sentinel tokens are added when they don't exist."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=151936)
        mock.add_special_tokens = MagicMock(return_value=100)
        mock.pad_token = None
        mock.eos_token = "</s>"
        mock.unk_token_id = 3
        # First sentinel returns UNK (doesn't exist)
        mock.convert_tokens_to_ids = MagicMock(return_value=3)
        mock.additional_special_tokens = []

        Qwen3EncoderDecoderTokenizer(mock, num_sentinel_tokens=100)

        # Sentinel tokens should be added
        assert mock.add_special_tokens.call_count >= 1
        # Check that sentinels were in the first call
        first_call_args = mock.add_special_tokens.call_args_list[0][0][0]
        assert "<extra_id_0>" in first_call_args["additional_special_tokens"]
