"""Extended tokenizer for Qwen3 Encoder-Decoder with sentinel tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Sentinel token format (following T5 convention)
SENTINEL_TOKEN_TEMPLATE = "<extra_id_{i}>"

# UL2 task prefix tokens
UL2_PREFIX_TOKENS = ["[R]", "[X]", "[S]", "[I]"]


class Qwen3EncoderDecoderTokenizer:
    """
    Wrapper tokenizer that extends Qwen3 tokenizer with sentinel tokens.

    This tokenizer adds sentinel tokens (<extra_id_0> through <extra_id_99>)
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
        original_vocab_size: Vocabulary size before adding sentinels.
    """

    def __init__(
        self,
        base_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        num_sentinel_tokens: int = 100,
        _original_vocab_size: Optional[int] = None,
    ) -> None:
        self.base_tokenizer = base_tokenizer
        self.num_sentinel_tokens = num_sentinel_tokens

        # Store original vocab size before adding sentinels
        # If provided (from sentinel_config.json), use that value
        # Otherwise compute from current tokenizer length
        self.original_vocab_size = (
            _original_vocab_size
            if _original_vocab_size is not None
            else len(base_tokenizer)
        )

        # Add sentinel tokens
        self._add_sentinel_tokens()

        # Add UL2 prefix tokens as special tokens
        self._add_ul2_prefix_tokens()

        # Create lookup dicts
        self._sentinel_tokens: Dict[int, str] = {
            i: SENTINEL_TOKEN_TEMPLATE.format(i=i) for i in range(num_sentinel_tokens)
        }
        self._sentinel_token_ids: Dict[int, int] = {
            i: self.original_vocab_size + i for i in range(num_sentinel_tokens)
        }

        # Reverse lookup
        self._id_to_sentinel_index: Dict[int, int] = {
            v: k for k, v in self._sentinel_token_ids.items()
        }

        # Ensure pad_token is set (Qwen3 may not have one)
        if self.base_tokenizer.pad_token is None:
            if self.base_tokenizer.eos_token is not None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
                logger.info("Set pad_token to eos_token since pad_token was not set")

    def _add_sentinel_tokens(self) -> None:
        """Add sentinel tokens to the tokenizer vocabulary."""
        if self.num_sentinel_tokens == 0:
            return

        sentinel_tokens = [
            SENTINEL_TOKEN_TEMPLATE.format(i=i) for i in range(self.num_sentinel_tokens)
        ]

        # Check if first sentinel already exists (indicates previously saved tokenizer)
        first_sentinel = sentinel_tokens[0]
        first_sentinel_id = self.base_tokenizer.convert_tokens_to_ids(first_sentinel)
        unk_id = self.base_tokenizer.unk_token_id

        # If first sentinel exists and isn't UNK, tokens were already added
        if first_sentinel_id != unk_id:
            logger.info(
                f"Sentinel tokens already exist in vocabulary (first sentinel ID: {first_sentinel_id})"
            )
            return

        # Add as special tokens
        num_added = self.base_tokenizer.add_special_tokens(
            {"additional_special_tokens": sentinel_tokens}
        )

        if num_added != self.num_sentinel_tokens:
            logger.warning(
                f"Expected to add {self.num_sentinel_tokens} sentinel tokens, "
                f"but added {num_added}. Some tokens may already exist."
            )
        else:
            logger.info(f"Added {num_added} sentinel tokens to vocabulary")

    def _add_ul2_prefix_tokens(self) -> None:
        """Add UL2 task prefix tokens ([R], [X], [S], [I]) as special tokens."""
        # Check if first prefix already exists
        first_prefix = UL2_PREFIX_TOKENS[0]
        first_prefix_id = self.base_tokenizer.convert_tokens_to_ids(first_prefix)
        unk_id = self.base_tokenizer.unk_token_id

        if first_prefix_id != unk_id:
            logger.info(
                f"UL2 prefix tokens already exist in vocabulary (first prefix ID: {first_prefix_id})"
            )
            return

        # Get existing additional_special_tokens to avoid overwriting
        existing = self.base_tokenizer.additional_special_tokens or []
        new_tokens = [t for t in UL2_PREFIX_TOKENS if t not in existing]

        if new_tokens:
            num_added = self.base_tokenizer.add_special_tokens(
                {"additional_special_tokens": existing + new_tokens}
            )
            if num_added > 0:
                logger.info(f"Added {num_added} UL2 prefix tokens: {new_tokens}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_sentinel_tokens: int = 100,
        **kwargs: Any,
    ) -> Qwen3EncoderDecoderTokenizer:
        """
        Load tokenizer from pretrained Qwen3 checkpoint or saved directory.

        If loading from a directory that contains sentinel_config.json,
        the sentinel configuration will be restored from that file.

        Args:
            pretrained_model_name_or_path: Path or HuggingFace Hub ID.
            num_sentinel_tokens: Number of sentinel tokens to add.
                Ignored if sentinel_config.json exists.
            **kwargs: Additional arguments for AutoTokenizer.

        Returns:
            Qwen3EncoderDecoderTokenizer instance.
        """
        path = Path(pretrained_model_name_or_path)

        # Check if loading from a previously saved tokenizer with sentinel config
        sentinel_config_path = path / "sentinel_config.json" if path.exists() else None
        original_vocab_size: Optional[int] = None

        if sentinel_config_path and sentinel_config_path.exists():
            with open(sentinel_config_path) as f:
                sentinel_config = json.load(f)
            num_sentinel_tokens = sentinel_config.get(
                "num_sentinel_tokens", num_sentinel_tokens
            )
            original_vocab_size = sentinel_config.get("original_vocab_size")
            logger.info(
                f"Loaded sentinel config: {num_sentinel_tokens} sentinel tokens, "
                f"original_vocab_size={original_vocab_size}"
            )

        base_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        return cls(
            base_tokenizer,
            num_sentinel_tokens=num_sentinel_tokens,
            _original_vocab_size=original_vocab_size,
        )

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save tokenizer to directory.

        Saves both the base tokenizer files and a sentinel_config.json
        containing the sentinel token configuration.

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

        logger.info(f"Saved tokenizer with {self.num_sentinel_tokens} sentinel tokens")

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
        return self._sentinel_tokens[index]

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
        return self._sentinel_token_ids[index]

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
        end: Optional[int] = None,
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
    # Properties
    # =========================================================================

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
    # Delegation to Base Tokenizer
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base tokenizer."""
        # Avoid infinite recursion for base_tokenizer itself
        if name in (
            "base_tokenizer",
            "_sentinel_tokens",
            "_sentinel_token_ids",
            "_id_to_sentinel_index",
        ):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.base_tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate tokenization to base tokenizer."""
        return self.base_tokenizer(*args, **kwargs)

    def __len__(self) -> int:
        """Return vocabulary size including sentinels."""
        return len(self.base_tokenizer)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"num_sentinel_tokens={self.num_sentinel_tokens}, "
            f"original_vocab_size={self.original_vocab_size})"
        )


# =============================================================================
# Utility Functions for Span Corruption
# =============================================================================


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
        - target_ids: Flattened list of [sentinel_0, span_0_tokens, ...]

    Example:
        >>> spans = [[101, 102], [201, 202, 203]]  # Two corrupted spans
        >>> sentinel_ids, target_ids = create_sentinel_sequence(tokenizer, spans)
        >>> # sentinel_ids = [151936, 151937]  # <extra_id_0>, <extra_id_1>
        >>> # target_ids = [151936, 101, 102, 151937, 201, 202, 203]

    Raises:
        ValueError: If number of spans exceeds available sentinel tokens.
    """
    if len(spans) > tokenizer.num_sentinel_tokens:
        raise ValueError(
            f"Number of spans ({len(spans)}) exceeds number of "
            f"sentinel tokens ({tokenizer.num_sentinel_tokens})"
        )

    sentinel_ids = [tokenizer.get_sentinel_token_id(i) for i in range(len(spans))]

    # Build target sequence: [<s0>, span0, <s1>, span1, ...]
    target_ids: List[int] = []
    for i, span in enumerate(spans):
        target_ids.append(sentinel_ids[i])
        target_ids.extend(span)

    return sentinel_ids, target_ids


def apply_sentinel_corruption(
    input_ids: List[int],
    spans_to_corrupt: List[Tuple[int, int]],
    tokenizer: Qwen3EncoderDecoderTokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Apply sentinel-based corruption to a token sequence.

    Args:
        input_ids: Original token IDs.
        spans_to_corrupt: List of (start, end) tuples indicating which
            positions to replace with sentinels. Positions are inclusive
            for start, exclusive for end.
        tokenizer: The tokenizer instance.

    Returns:
        Tuple of (encoder_input_ids, decoder_target_ids).

    Example:
        >>> input_ids = [10, 20, 30, 40, 50, 60, 70]
        >>> spans = [(1, 3), (5, 6)]  # Replace positions 1-2 and 5
        >>> enc_ids, dec_ids = apply_sentinel_corruption(input_ids, spans, tokenizer)
        >>> # enc_ids = [10, <s0>, 40, 50, <s1>, 70]
        >>> # dec_ids = [<s0>, 20, 30, <s1>, 60]

    Raises:
        ValueError: If number of spans exceeds available sentinel tokens.
    """
    if len(spans_to_corrupt) > tokenizer.num_sentinel_tokens:
        raise ValueError(
            f"Number of spans ({len(spans_to_corrupt)}) exceeds "
            f"sentinel tokens ({tokenizer.num_sentinel_tokens})"
        )

    # Sort spans by start position
    sorted_spans = sorted(spans_to_corrupt, key=lambda x: x[0])

    # Extract corrupted spans
    corrupted_spans: List[List[int]] = []
    for start, end in sorted_spans:
        corrupted_spans.append(input_ids[start:end])

    # Build encoder input (replace spans with sentinels)
    encoder_input: List[int] = []
    prev_end = 0

    for i, (start, end) in enumerate(sorted_spans):
        # Add tokens before this span
        encoder_input.extend(input_ids[prev_end:start])
        # Add sentinel token
        encoder_input.append(tokenizer.get_sentinel_token_id(i))
        prev_end = end

    # Add remaining tokens after last span
    encoder_input.extend(input_ids[prev_end:])

    # Build decoder target
    _, decoder_target = create_sentinel_sequence(tokenizer, corrupted_spans)

    return encoder_input, decoder_target
