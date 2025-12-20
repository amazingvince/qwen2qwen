# Story 02: Tokenizer Extension with Sentinel Tokens

## Overview

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-002 |
| **Title** | Tokenizer Extension with Sentinel Tokens |
| **Priority** | P0 - Critical Path |
| **Estimated Effort** | 1-2 days |
| **Dependencies** | Story 01 (Configuration Class) |
| **Deliverables** | Extended tokenizer class, sentinel token handling, unit tests |

---

## Objective

Extend the Qwen3 tokenizer to support sentinel tokens required for UL2 span corruption training. Sentinel tokens (e.g., `<extra_id_0>`, `<extra_id_1>`, ..., `<extra_id_99>`) are special placeholder tokens that mark spans to be predicted by the decoder.

---

## Background & Context

### What Are Sentinel Tokens?
In UL2/T5-style training, input text is corrupted by replacing spans with sentinel tokens:

**Original text:**
```
The quick brown fox jumps over the lazy dog
```

**After span corruption (input to encoder):**
```
The quick <extra_id_0> jumps over <extra_id_1> dog
```

**Target (for decoder):**
```
<extra_id_0> brown fox <extra_id_1> the lazy
```

### Why 100 Sentinels?
- T5 and UL2 use 100 sentinel tokens by convention
- This allows up to 100 separate spans per sequence
- In practice, most sequences have 5-20 spans depending on corruption rate

### Reference Implementations
1. **T5 Tokenizer**: `transformers/models/t5/tokenization_t5.py`
2. **Qwen3 Tokenizer**: Uses `tiktoken` via `Qwen2Tokenizer` base

---

## Technical Requirements

### 1. Tokenizer Class

#### File: `tokenization_qwen3_encdec.py`

```python
"""Extended tokenizer for Qwen3 Encoder-Decoder with sentinel tokens."""

from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Sentinel token format (following T5 convention)
SENTINEL_TOKEN_TEMPLATE = "<extra_id_{i}>"


class Qwen3EncoderDecoderTokenizer:
    """
    Wrapper tokenizer that extends Qwen3 tokenizer with sentinel tokens.
    
    This tokenizer adds 100 sentinel tokens (<extra_id_0> through <extra_id_99>)
    to the Qwen3 vocabulary for use in UL2-style span corruption training.
    
    Args:
        base_tokenizer: The underlying Qwen3 tokenizer instance.
        num_sentinel_tokens: Number of sentinel tokens to add (default: 100).
        
    Example:
        ```python
        from qwen3_encdec import Qwen3EncoderDecoderTokenizer
        
        # Load from base Qwen3 tokenizer
        tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        # Encode with sentinel tokens
        text = "The quick <extra_id_0> jumps over <extra_id_1> dog"
        tokens = tokenizer.encode(text)
        ```
    
    Attributes:
        base_tokenizer: The underlying Qwen3 tokenizer.
        num_sentinel_tokens: Number of sentinel tokens added.
        sentinel_token_ids: Dict mapping sentinel index to token ID.
        sentinel_tokens: Dict mapping sentinel index to token string.
    """
    
    def __init__(
        self,
        base_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        num_sentinel_tokens: int = 100,
    ):
        self.base_tokenizer = base_tokenizer
        self.num_sentinel_tokens = num_sentinel_tokens
        
        # Store original vocab size before adding sentinels
        self.original_vocab_size = len(base_tokenizer)
        
        # Add sentinel tokens
        self._add_sentinel_tokens()
        
        # Create lookup dicts
        self.sentinel_tokens = {
            i: SENTINEL_TOKEN_TEMPLATE.format(i=i)
            for i in range(num_sentinel_tokens)
        }
        self.sentinel_token_ids = {
            i: self.original_vocab_size + i
            for i in range(num_sentinel_tokens)
        }
        
        # Reverse lookup
        self._id_to_sentinel_index = {
            v: k for k, v in self.sentinel_token_ids.items()
        }
    
    def _add_sentinel_tokens(self):
        """Add sentinel tokens to the tokenizer vocabulary."""
        sentinel_tokens = [
            SENTINEL_TOKEN_TEMPLATE.format(i=i)
            for i in range(self.num_sentinel_tokens)
        ]
        
        # Add as special tokens
        num_added = self.base_tokenizer.add_special_tokens({
            "additional_special_tokens": sentinel_tokens
        })
        
        if num_added != self.num_sentinel_tokens:
            logger.warning(
                f"Expected to add {self.num_sentinel_tokens} sentinel tokens, "
                f"but added {num_added}. Some tokens may already exist."
            )
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_sentinel_tokens: int = 100,
        **kwargs
    ) -> "Qwen3EncoderDecoderTokenizer":
        """
        Load tokenizer from pretrained Qwen3 checkpoint.
        
        Args:
            pretrained_model_name_or_path: Path or HuggingFace Hub ID.
            num_sentinel_tokens: Number of sentinel tokens to add.
            **kwargs: Additional arguments for AutoTokenizer.
            
        Returns:
            Qwen3EncoderDecoderTokenizer instance.
        """
        base_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        return cls(base_tokenizer, num_sentinel_tokens=num_sentinel_tokens)
    
    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer to directory.
        
        Args:
            save_directory: Path to save directory.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save base tokenizer
        self.base_tokenizer.save_pretrained(save_directory)
        
        # Save sentinel config
        sentinel_config = {
            "num_sentinel_tokens": self.num_sentinel_tokens,
            "original_vocab_size": self.original_vocab_size,
            "sentinel_token_template": SENTINEL_TOKEN_TEMPLATE,
        }
        
        config_path = save_path / "sentinel_config.json"
        with open(config_path, "w") as f:
            json.dump(sentinel_config, f, indent=2)
    
    # =========================================================================
    # Sentinel Token Methods
    # =========================================================================
    
    def get_sentinel_token(self, index: int) -> str:
        """
        Get sentinel token string by index.
        
        Args:
            index: Sentinel index (0 to num_sentinel_tokens - 1).
            
        Returns:
            Sentinel token string (e.g., "<extra_id_0>").
            
        Raises:
            ValueError: If index is out of range.
        """
        if not 0 <= index < self.num_sentinel_tokens:
            raise ValueError(
                f"Sentinel index must be between 0 and {self.num_sentinel_tokens - 1}, "
                f"got {index}"
            )
        return self.sentinel_tokens[index]
    
    def get_sentinel_token_id(self, index: int) -> int:
        """
        Get sentinel token ID by index.
        
        Args:
            index: Sentinel index (0 to num_sentinel_tokens - 1).
            
        Returns:
            Token ID for the sentinel token.
            
        Raises:
            ValueError: If index is out of range.
        """
        if not 0 <= index < self.num_sentinel_tokens:
            raise ValueError(
                f"Sentinel index must be between 0 and {self.num_sentinel_tokens - 1}, "
                f"got {index}"
            )
        return self.sentinel_token_ids[index]
    
    def is_sentinel_token_id(self, token_id: int) -> bool:
        """
        Check if a token ID is a sentinel token.
        
        Args:
            token_id: Token ID to check.
            
        Returns:
            True if token is a sentinel token.
        """
        return token_id in self._id_to_sentinel_index
    
    def sentinel_id_to_index(self, token_id: int) -> int:
        """
        Convert sentinel token ID to sentinel index.
        
        Args:
            token_id: Sentinel token ID.
            
        Returns:
            Sentinel index (0 to num_sentinel_tokens - 1).
            
        Raises:
            ValueError: If token_id is not a sentinel token.
        """
        if token_id not in self._id_to_sentinel_index:
            raise ValueError(f"Token ID {token_id} is not a sentinel token")
        return self._id_to_sentinel_index[token_id]
    
    def get_sentinel_tokens_in_range(
        self, 
        start: int = 0, 
        end: Optional[int] = None
    ) -> List[str]:
        """
        Get a range of sentinel tokens.
        
        Args:
            start: Starting index (inclusive).
            end: Ending index (exclusive). Defaults to num_sentinel_tokens.
            
        Returns:
            List of sentinel token strings.
        """
        if end is None:
            end = self.num_sentinel_tokens
        return [self.get_sentinel_token(i) for i in range(start, end)]
    
    # =========================================================================
    # Delegation to Base Tokenizer
    # =========================================================================
    
    def __call__(self, *args, **kwargs):
        """Delegate tokenization to base tokenizer."""
        return self.base_tokenizer(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        """Delegate encoding to base tokenizer."""
        return self.base_tokenizer.encode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """Delegate decoding to base tokenizer."""
        return self.base_tokenizer.decode(*args, **kwargs)
    
    def encode_plus(self, *args, **kwargs):
        """Delegate encode_plus to base tokenizer."""
        return self.base_tokenizer.encode_plus(*args, **kwargs)
    
    def batch_encode_plus(self, *args, **kwargs):
        """Delegate batch_encode_plus to base tokenizer."""
        return self.base_tokenizer.batch_encode_plus(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        """Delegate batch_decode to base tokenizer."""
        return self.base_tokenizer.batch_decode(*args, **kwargs)
    
    def convert_tokens_to_ids(self, tokens):
        """Delegate token to ID conversion to base tokenizer."""
        return self.base_tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        """Delegate ID to token conversion to base tokenizer."""
        return self.base_tokenizer.convert_ids_to_tokens(ids)
    
    def __len__(self):
        """Return vocabulary size including sentinels."""
        return len(self.base_tokenizer)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including sentinels."""
        return len(self.base_tokenizer)
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Return pad token ID."""
        return self.base_tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Return EOS token ID."""
        return self.base_tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Return BOS token ID."""
        return self.base_tokenizer.bos_token_id
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """Return UNK token ID."""
        return self.base_tokenizer.unk_token_id
    
    # =========================================================================
    # Encoder-Decoder Specific Methods
    # =========================================================================
    
    def build_inputs_with_special_tokens(
        self,
        encoder_input_ids: List[int],
        decoder_input_ids: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Build model inputs from encoder and decoder sequences.
        
        For encoder-decoder models, this adds appropriate special tokens
        to both sequences.
        
        Args:
            encoder_input_ids: Encoder input token IDs.
            decoder_input_ids: Decoder input token IDs (optional).
            
        Returns:
            Tuple of (encoder_input_ids, decoder_input_ids) with special tokens.
        """
        # Add EOS to encoder if configured
        if self.eos_token_id is not None:
            if len(encoder_input_ids) == 0 or encoder_input_ids[-1] != self.eos_token_id:
                encoder_input_ids = encoder_input_ids + [self.eos_token_id]
        
        if decoder_input_ids is not None:
            # Add BOS to decoder if configured
            if self.bos_token_id is not None:
                if len(decoder_input_ids) == 0 or decoder_input_ids[0] != self.bos_token_id:
                    decoder_input_ids = [self.bos_token_id] + decoder_input_ids
        
        return encoder_input_ids, decoder_input_ids
    
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """
        Prepare a batch for sequence-to-sequence training.
        
        Args:
            src_texts: Source (encoder input) texts.
            tgt_texts: Target (decoder output) texts (optional).
            max_length: Maximum encoder sequence length.
            max_target_length: Maximum decoder sequence length.
            padding: Padding strategy ("longest", "max_length", or False).
            truncation: Whether to truncate sequences.
            return_tensors: Output tensor type ("pt", "tf", "np", or None).
            
        Returns:
            Dictionary with input_ids, attention_mask, labels, etc.
        """
        # Encode source texts
        model_inputs = self.base_tokenizer(
            src_texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        
        # Encode target texts if provided
        if tgt_texts is not None:
            with self.base_tokenizer.as_target_tokenizer():
                labels = self.base_tokenizer(
                    tgt_texts,
                    max_length=max_target_length,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors,
                )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # Replace padding token id with -100 for loss computation
            if self.pad_token_id is not None and return_tensors == "pt":
                import torch
                labels_tensor = model_inputs["labels"]
                labels_tensor[labels_tensor == self.pad_token_id] = -100
                model_inputs["labels"] = labels_tensor
        
        return model_inputs
```

### 2. Helper Functions for Span Corruption

Add these utility functions for working with sentinels during data processing:

```python
# Additional utility functions in tokenization_qwen3_encdec.py

def create_sentinel_sequence(
    tokenizer: Qwen3EncoderDecoderTokenizer,
    spans: List[List[int]],
) -> Tuple[List[int], List[int]]:
    """
    Create encoder input and decoder target from span information.
    
    This is a utility function for UL2-style span corruption.
    
    Args:
        tokenizer: The tokenizer instance.
        spans: List of spans, where each span is a list of token IDs.
               These are the tokens that were replaced with sentinels.
               
    Returns:
        Tuple of (sentinel_ids, target_ids) where:
        - sentinel_ids: List of sentinel token IDs to use as placeholders
        - target_ids: Flattened list of [sentinel_0, span_0_tokens, sentinel_1, span_1_tokens, ...]
        
    Example:
        >>> spans = [[101, 102], [201, 202, 203]]  # Two corrupted spans
        >>> sentinel_ids, target_ids = create_sentinel_sequence(tokenizer, spans)
        >>> # sentinel_ids = [151936, 151937]  # <extra_id_0>, <extra_id_1>
        >>> # target_ids = [151936, 101, 102, 151937, 201, 202, 203]
    """
    if len(spans) > tokenizer.num_sentinel_tokens:
        raise ValueError(
            f"Number of spans ({len(spans)}) exceeds number of "
            f"sentinel tokens ({tokenizer.num_sentinel_tokens})"
        )
    
    sentinel_ids = [tokenizer.get_sentinel_token_id(i) for i in range(len(spans))]
    
    # Build target sequence: [<s0>, span0, <s1>, span1, ...]
    target_ids = []
    for i, span in enumerate(spans):
        target_ids.append(sentinel_ids[i])
        target_ids.extend(span)
    
    return sentinel_ids, target_ids


def apply_sentinel_corruption(
    input_ids: List[int],
    spans_to_corrupt: List[Tuple[int, int]],  # (start, end) positions
    tokenizer: Qwen3EncoderDecoderTokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Apply sentinel-based corruption to a token sequence.
    
    Args:
        input_ids: Original token IDs.
        spans_to_corrupt: List of (start, end) tuples indicating which
                         positions to replace with sentinels.
        tokenizer: The tokenizer instance.
        
    Returns:
        Tuple of (encoder_input_ids, decoder_target_ids).
        
    Example:
        >>> input_ids = [10, 20, 30, 40, 50, 60, 70]
        >>> spans = [(1, 3), (5, 6)]  # Replace positions 1-2 and 5
        >>> enc_ids, dec_ids = apply_sentinel_corruption(input_ids, spans, tokenizer)
        >>> # enc_ids = [10, <s0>, 40, 50, <s1>, 70]
        >>> # dec_ids = [<s0>, 20, 30, <s1>, 60]
    """
    if len(spans_to_corrupt) > tokenizer.num_sentinel_tokens:
        raise ValueError(
            f"Number of spans ({len(spans_to_corrupt)}) exceeds "
            f"sentinel tokens ({tokenizer.num_sentinel_tokens})"
        )
    
    # Sort spans by start position (reverse for easier processing)
    sorted_spans = sorted(spans_to_corrupt, key=lambda x: x[0])
    
    # Extract corrupted spans
    corrupted_spans = []
    for start, end in sorted_spans:
        corrupted_spans.append(input_ids[start:end])
    
    # Build encoder input (replace spans with sentinels)
    encoder_input = []
    prev_end = 0
    
    for i, (start, end) in enumerate(sorted_spans):
        # Add tokens before this span
        encoder_input.extend(input_ids[prev_end:start])
        # Add sentinel token
        encoder_input.append(tokenizer.get_sentinel_token_id(i))
        prev_end = end
    
    # Add remaining tokens
    encoder_input.extend(input_ids[prev_end:])
    
    # Build decoder target
    _, decoder_target = create_sentinel_sequence(tokenizer, corrupted_spans)
    
    return encoder_input, decoder_target
```

---

## Unit Tests

#### File: `tests/test_tokenization.py`

```python
"""Unit tests for Qwen3EncoderDecoderTokenizer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen3_encdec.tokenization_qwen3_encdec import (
    Qwen3EncoderDecoderTokenizer,
    SENTINEL_TOKEN_TEMPLATE,
    create_sentinel_sequence,
    apply_sentinel_corruption,
)


class TestQwen3EncoderDecoderTokenizer:
    """Test suite for extended tokenizer."""
    
    @pytest.fixture
    def mock_base_tokenizer(self):
        """Create a mock base tokenizer for testing."""
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=151936)
        mock.add_special_tokens = MagicMock(return_value=100)
        mock.pad_token_id = 0
        mock.eos_token_id = 1
        mock.bos_token_id = 2
        mock.unk_token_id = 3
        return mock
    
    @pytest.fixture
    def tokenizer(self, mock_base_tokenizer):
        """Create tokenizer instance for testing."""
        return Qwen3EncoderDecoderTokenizer(
            mock_base_tokenizer,
            num_sentinel_tokens=100
        )
    
    def test_initialization(self, tokenizer, mock_base_tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer.num_sentinel_tokens == 100
        assert tokenizer.original_vocab_size == 151936
        
        # Verify sentinel tokens were added
        mock_base_tokenizer.add_special_tokens.assert_called_once()
        call_args = mock_base_tokenizer.add_special_tokens.call_args
        added_tokens = call_args[0][0]["additional_special_tokens"]
        assert len(added_tokens) == 100
        assert added_tokens[0] == "<extra_id_0>"
        assert added_tokens[99] == "<extra_id_99>"
    
    def test_get_sentinel_token(self, tokenizer):
        """Test sentinel token string retrieval."""
        assert tokenizer.get_sentinel_token(0) == "<extra_id_0>"
        assert tokenizer.get_sentinel_token(50) == "<extra_id_50>"
        assert tokenizer.get_sentinel_token(99) == "<extra_id_99>"
    
    def test_get_sentinel_token_out_of_range(self, tokenizer):
        """Test error on out of range sentinel index."""
        with pytest.raises(ValueError):
            tokenizer.get_sentinel_token(-1)
        
        with pytest.raises(ValueError):
            tokenizer.get_sentinel_token(100)
    
    def test_get_sentinel_token_id(self, tokenizer):
        """Test sentinel token ID retrieval."""
        assert tokenizer.get_sentinel_token_id(0) == 151936
        assert tokenizer.get_sentinel_token_id(1) == 151937
        assert tokenizer.get_sentinel_token_id(99) == 152035
    
    def test_is_sentinel_token_id(self, tokenizer):
        """Test sentinel token ID detection."""
        # Regular tokens
        assert not tokenizer.is_sentinel_token_id(0)
        assert not tokenizer.is_sentinel_token_id(151935)
        
        # Sentinel tokens
        assert tokenizer.is_sentinel_token_id(151936)
        assert tokenizer.is_sentinel_token_id(152035)
    
    def test_sentinel_id_to_index(self, tokenizer):
        """Test converting sentinel ID to index."""
        assert tokenizer.sentinel_id_to_index(151936) == 0
        assert tokenizer.sentinel_id_to_index(151937) == 1
        assert tokenizer.sentinel_id_to_index(152035) == 99
    
    def test_sentinel_id_to_index_invalid(self, tokenizer):
        """Test error on non-sentinel token ID."""
        with pytest.raises(ValueError):
            tokenizer.sentinel_id_to_index(100)  # Regular token
    
    def test_get_sentinel_tokens_in_range(self, tokenizer):
        """Test getting range of sentinel tokens."""
        tokens = tokenizer.get_sentinel_tokens_in_range(0, 3)
        assert tokens == ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"]
        
        tokens = tokenizer.get_sentinel_tokens_in_range(97)
        assert tokens == ["<extra_id_97>", "<extra_id_98>", "<extra_id_99>"]
    
    def test_delegation_to_base_tokenizer(self, tokenizer, mock_base_tokenizer):
        """Test that methods are properly delegated."""
        # Test encode
        tokenizer.encode("test")
        mock_base_tokenizer.encode.assert_called_once_with("test")
        
        # Test decode
        mock_base_tokenizer.reset_mock()
        tokenizer.decode([1, 2, 3])
        mock_base_tokenizer.decode.assert_called_once_with([1, 2, 3])
    
    def test_vocab_size_property(self, tokenizer, mock_base_tokenizer):
        """Test vocab_size property."""
        assert tokenizer.vocab_size == len(mock_base_tokenizer)
    
    def test_special_token_properties(self, tokenizer):
        """Test special token ID properties."""
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.unk_token_id == 3


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
    
    def test_create_sentinel_sequence_too_many_spans(self, mock_tokenizer):
        """Test error when too many spans."""
        spans = [[i] for i in range(101)]  # 101 spans
        
        with pytest.raises(ValueError, match="exceeds"):
            create_sentinel_sequence(mock_tokenizer, spans)
    
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


class TestTokenizerSaveLoad:
    """Test tokenizer serialization."""
    
    @patch("qwen3_encdec.tokenization_qwen3_encdec.AutoTokenizer")
    def test_from_pretrained(self, mock_auto_tokenizer):
        """Test loading from pretrained."""
        mock_base = MagicMock()
        mock_base.__len__ = MagicMock(return_value=151936)
        mock_base.add_special_tokens = MagicMock(return_value=100)
        mock_auto_tokenizer.from_pretrained.return_value = mock_base
        
        tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            num_sentinel_tokens=50
        )
        
        assert tokenizer.num_sentinel_tokens == 50
        mock_auto_tokenizer.from_pretrained.assert_called_once()
    
    def test_save_pretrained(self):
        """Test saving tokenizer."""
        mock_base = MagicMock()
        mock_base.__len__ = MagicMock(return_value=151936)
        mock_base.add_special_tokens = MagicMock(return_value=100)
        
        tokenizer = Qwen3EncoderDecoderTokenizer(mock_base, num_sentinel_tokens=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            
            # Verify base tokenizer save was called
            mock_base.save_pretrained.assert_called_once_with(tmpdir)
            
            # Verify sentinel config was saved
            config_path = Path(tmpdir) / "sentinel_config.json"
            assert config_path.exists()
            
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            assert config["num_sentinel_tokens"] == 100
            assert config["original_vocab_size"] == 151936
```

---

## Integration Test with Real Tokenizer

Create an integration test that uses the actual Qwen3 tokenizer:

```python
# tests/test_tokenization_integration.py
"""Integration tests with real Qwen3 tokenizer."""

import pytest
from transformers import AutoTokenizer

# Skip if Qwen3 not available
pytest.importorskip("transformers")


class TestRealTokenizerIntegration:
    """Integration tests with actual Qwen3 tokenizer."""
    
    @pytest.fixture(scope="class")
    def real_tokenizer(self):
        """Load real Qwen3 tokenizer (cached at class level)."""
        try:
            from qwen3_encdec.tokenization_qwen3_encdec import (
                Qwen3EncoderDecoderTokenizer
            )
            return Qwen3EncoderDecoderTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B",
                num_sentinel_tokens=100
            )
        except Exception as e:
            pytest.skip(f"Could not load Qwen3 tokenizer: {e}")
    
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
        sentinel_0_id = real_tokenizer.get_sentinel_token_id(0)
        tokens = [100, 200, sentinel_0_id, 300]  # Example token IDs
        
        decoded = real_tokenizer.decode(tokens)
        assert "<extra_id_0>" in decoded
    
    def test_roundtrip(self, real_tokenizer):
        """Test encode-decode roundtrip."""
        original = "Hello <extra_id_0> world"
        encoded = real_tokenizer.encode(original)
        decoded = real_tokenizer.decode(encoded)
        
        # Decoded should contain original content
        assert "<extra_id_0>" in decoded
```

---

## Acceptance Criteria

1. **Sentinel Token Addition**: 100 sentinel tokens are added to vocabulary
2. **Token ID Mapping**: Sentinel IDs start at original_vocab_size
3. **String Retrieval**: `get_sentinel_token(i)` returns correct string
4. **ID Retrieval**: `get_sentinel_token_id(i)` returns correct ID
5. **Detection**: `is_sentinel_token_id()` correctly identifies sentinels
6. **Corruption Utilities**: Helper functions work correctly
7. **Serialization**: Save/load preserves sentinel configuration
8. **Delegation**: Base tokenizer methods work through wrapper
9. **Unit Tests**: All tests pass with >95% coverage
10. **Integration Test**: Works with real Qwen3 tokenizer

---

## Dependencies

```txt
# requirements.txt (additions)
tiktoken>=0.5.0  # Qwen3 tokenizer backend
sentencepiece>=0.1.99  # Fallback tokenizer support
```

---

## Notes for Developer

1. **Qwen3 Tokenizer Type**: Qwen3 uses a tiktoken-based fast tokenizer. Verify the exact class hierarchy when implementing.

2. **Special Token Handling**: Some tokenizers have specific methods for adding special tokens. Test with the real tokenizer early.

3. **Context Manager**: The `as_target_tokenizer()` context manager may not be implemented in all tokenizers - add fallback if needed.

4. **Padding Token**: Qwen3 may not have a default pad token. You may need to set `tokenizer.pad_token = tokenizer.eos_token`.

5. **Vocab Size Verification**: After adding sentinels, verify `len(tokenizer)` equals expected value.

---

## Next Story

After completing this story, proceed to **Story 03: Qwen3 Encoder Implementation (Bidirectional)**.
