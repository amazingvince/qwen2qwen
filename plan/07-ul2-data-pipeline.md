# Story 07: UL2 Data Pipeline

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-07 |
| **Title** | UL2 Mixture-of-Denoisers Data Pipeline |
| **Epic** | Qwen3 Encoder-Decoder Implementation |
| **Priority** | High |
| **Estimated Effort** | 3-4 days |
| **Dependencies** | Story 02 (Tokenizer with Sentinel Tokens) |

---

## Objective

Implement the UL2 (Unified Language Learner) data pipeline with five denoising tasks following T5Gemma 2's training approach. The pipeline should efficiently corrupt text, apply sentinel tokens, and produce encoder-decoder training examples with streaming support for large-scale datasets.

---

## Background

### UL2 Training Objective

UL2 uses a Mixture-of-Denoisers approach with three types of tasks:

1. **R-Denoiser (Regular)**: Short spans, low corruption - like BERT's MLM
2. **X-Denoiser (Extreme)**: Long spans, variable corruption - for in-context learning
3. **S-Denoiser (Sequential)**: Prefix-to-suffix - like GPT-style generation

### T5Gemma 2 Task Configuration

| Task | Mean Span (μ) | Corruption Rate (r) | Mix Weight |
|------|---------------|---------------------|------------|
| R-Denoiser 1 | 3 | 0.15 | 1 |
| R-Denoiser 2 | 12 | 0.50 | 1 |
| X-Denoiser 1 | 32 | 0.15 | 1 |
| X-Denoiser 2 | 32 | 0.50 | 1 |
| S-Denoiser | 0.75×L | 0.75 | **4** |

The S-Denoiser has 4× weight because sequential/causal generation is crucial for downstream tasks.

---

## Technical Requirements

### 7.1 UL2 Task Implementation

```python
# src/data/ul2_corruption.py

"""
UL2 Mixture-of-Denoisers corruption utilities.

Implements the five denoising tasks from T5Gemma 2:
- R-Denoiser 1 & 2: Regular span corruption
- X-Denoiser 1 & 2: Extreme span corruption  
- S-Denoiser: Sequential prefix-to-suffix
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class UL2TaskType(Enum):
    """UL2 task types."""
    R_DENOISER_1 = "r1"  # μ=3, r=0.15
    R_DENOISER_2 = "r2"  # μ=12, r=0.50
    X_DENOISER_1 = "x1"  # μ=32, r=0.15
    X_DENOISER_2 = "x2"  # μ=32, r=0.50
    S_DENOISER = "s"     # μ=0.75*L, r=0.75


@dataclass
class UL2TaskConfig:
    """Configuration for a single UL2 task."""
    task_type: UL2TaskType
    mean_span_length: Optional[float]  # None for S-Denoiser (computed from length)
    corruption_rate: float
    mix_weight: int = 1
    
    # For S-Denoiser, this is the fraction of sequence to corrupt
    length_fraction: Optional[float] = None


# Default T5Gemma 2 task configurations
DEFAULT_UL2_TASKS = [
    UL2TaskConfig(UL2TaskType.R_DENOISER_1, mean_span_length=3, corruption_rate=0.15, mix_weight=1),
    UL2TaskConfig(UL2TaskType.R_DENOISER_2, mean_span_length=12, corruption_rate=0.50, mix_weight=1),
    UL2TaskConfig(UL2TaskType.X_DENOISER_1, mean_span_length=32, corruption_rate=0.15, mix_weight=1),
    UL2TaskConfig(UL2TaskType.X_DENOISER_2, mean_span_length=32, corruption_rate=0.50, mix_weight=1),
    UL2TaskConfig(UL2TaskType.S_DENOISER, mean_span_length=None, corruption_rate=0.75, 
                  mix_weight=4, length_fraction=0.75),
]


class UL2TaskSampler:
    """
    Samples UL2 tasks according to mixing weights.
    
    Default weights: 1:1:1:1:4 for r1:r2:x1:x2:s
    """
    
    def __init__(self, tasks: List[UL2TaskConfig] = None):
        """
        Initialize sampler.
        
        Args:
            tasks: List of task configurations. Defaults to T5Gemma 2 config.
        """
        self.tasks = tasks or DEFAULT_UL2_TASKS
        
        # Build sampling weights
        self.weights = [t.mix_weight for t in self.tasks]
        self.total_weight = sum(self.weights)
    
    def sample(self) -> UL2TaskConfig:
        """
        Sample a task according to mixing weights.
        
        Returns:
            Sampled task configuration
        """
        return random.choices(self.tasks, weights=self.weights, k=1)[0]
    
    def sample_batch(self, batch_size: int) -> List[UL2TaskConfig]:
        """
        Sample tasks for a batch.
        
        Args:
            batch_size: Number of tasks to sample
            
        Returns:
            List of task configurations
        """
        return random.choices(self.tasks, weights=self.weights, k=batch_size)


def compute_span_boundaries(
    sequence_length: int,
    mean_span_length: float,
    corruption_rate: float,
    random_state: Optional[np.random.RandomState] = None,
) -> List[Tuple[int, int]]:
    """
    Compute span boundaries for corruption.
    
    Uses geometric distribution for span lengths (like T5).
    
    Args:
        sequence_length: Length of input sequence
        mean_span_length: Mean span length (μ)
        corruption_rate: Fraction of tokens to corrupt (r)
        random_state: Optional random state for reproducibility
        
    Returns:
        List of (start, end) tuples for spans to corrupt
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Total tokens to corrupt
    num_tokens_to_corrupt = int(sequence_length * corruption_rate)
    
    if num_tokens_to_corrupt == 0:
        return []
    
    # Compute number of spans
    # Expected number of spans = num_tokens_to_corrupt / mean_span_length
    expected_num_spans = max(1, num_tokens_to_corrupt / mean_span_length)
    
    spans = []
    tokens_corrupted = 0
    
    # Sample span lengths from geometric distribution
    # P(length = k) = (1 - p)^(k-1) * p, where E[length] = 1/p
    p = 1.0 / mean_span_length
    
    while tokens_corrupted < num_tokens_to_corrupt:
        # Sample span length
        span_length = random_state.geometric(p)
        span_length = min(span_length, num_tokens_to_corrupt - tokens_corrupted)
        
        if span_length > 0:
            spans.append(span_length)
            tokens_corrupted += span_length
    
    # Distribute spans across the sequence
    # We need to place len(spans) spans such that they don't overlap
    num_spans = len(spans)
    
    if num_spans == 0:
        return []
    
    # Available positions for span starts
    # After placing spans, we need: sequence_length - sum(spans) positions for non-spans
    non_corrupted_length = sequence_length - sum(spans)
    
    if non_corrupted_length < num_spans + 1:
        # Not enough room, simplify to single span
        start = random_state.randint(0, max(1, sequence_length - sum(spans)))
        return [(start, start + sum(spans))]
    
    # Randomly place spans in available gaps
    # Create gap sizes (at least 1 token between spans)
    gaps = [1] * (num_spans + 1)  # Gaps before, between, and after spans
    remaining = non_corrupted_length - (num_spans + 1)
    
    # Distribute remaining tokens randomly to gaps
    for _ in range(remaining):
        gap_idx = random_state.randint(0, num_spans + 1)
        gaps[gap_idx] += 1
    
    # Build span boundaries
    span_boundaries = []
    position = gaps[0] - 1  # Adjust for 0-indexing
    
    for i, span_len in enumerate(spans):
        start = position
        end = start + span_len
        span_boundaries.append((start, end))
        position = end + gaps[i + 1]
    
    return span_boundaries


def apply_span_corruption(
    token_ids: List[int],
    span_boundaries: List[Tuple[int, int]],
    get_sentinel_id_fn,
) -> Tuple[List[int], List[int]]:
    """
    Apply span corruption to token sequence.
    
    Args:
        token_ids: Original token IDs
        span_boundaries: List of (start, end) span boundaries
        get_sentinel_id_fn: Function to get sentinel token ID (sentinel_idx -> token_id)
        
    Returns:
        Tuple of (encoder_input_ids, decoder_target_ids)
        
    Example:
        Input:  [A, B, C, D, E, F, G, H]
        Spans:  [(1, 3), (5, 7)]  # Corrupt B,C and F,G
        
        Encoder input:  [A, <s0>, D, E, <s1>, H]
        Decoder target: [<s0>, B, C, <s1>, F, G]
    """
    if not span_boundaries:
        # No corruption - return as-is (for edge cases)
        return token_ids.copy(), []
    
    # Sort spans by start position
    sorted_spans = sorted(span_boundaries, key=lambda x: x[0])
    
    encoder_ids = []
    decoder_ids = []
    
    prev_end = 0
    
    for sentinel_idx, (start, end) in enumerate(sorted_spans):
        # Add non-corrupted tokens before this span
        encoder_ids.extend(token_ids[prev_end:start])
        
        # Add sentinel to encoder input
        sentinel_id = get_sentinel_id_fn(sentinel_idx)
        encoder_ids.append(sentinel_id)
        
        # Add sentinel and corrupted tokens to decoder target
        decoder_ids.append(sentinel_id)
        decoder_ids.extend(token_ids[start:end])
        
        prev_end = end
    
    # Add remaining tokens after last span
    encoder_ids.extend(token_ids[prev_end:])
    
    return encoder_ids, decoder_ids


def apply_s_denoising(
    token_ids: List[int],
    length_fraction: float,
    get_sentinel_id_fn,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[List[int], List[int]]:
    """
    Apply S-Denoiser (sequential) corruption.
    
    Takes a prefix of the sequence and trains model to generate the suffix.
    
    Args:
        token_ids: Original token IDs
        length_fraction: Fraction of sequence to use as suffix (target)
        get_sentinel_id_fn: Function to get sentinel token ID
        random_state: Optional random state
        
    Returns:
        Tuple of (encoder_input_ids, decoder_target_ids)
        
    Example:
        Input: [A, B, C, D, E, F, G, H]
        length_fraction: 0.75 (6 tokens as suffix)
        
        Encoder input:  [A, B, <s0>]
        Decoder target: [<s0>, C, D, E, F, G, H]
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    sequence_length = len(token_ids)
    
    # Compute split point
    suffix_length = int(sequence_length * length_fraction)
    prefix_length = sequence_length - suffix_length
    
    # Ensure at least 1 token in each part
    prefix_length = max(1, min(prefix_length, sequence_length - 1))
    
    # Split
    prefix_ids = token_ids[:prefix_length]
    suffix_ids = token_ids[prefix_length:]
    
    # Add sentinel
    sentinel_id = get_sentinel_id_fn(0)
    
    encoder_ids = prefix_ids + [sentinel_id]
    decoder_ids = [sentinel_id] + suffix_ids
    
    return encoder_ids, decoder_ids


@dataclass
class UL2CorruptedExample:
    """Result of UL2 corruption."""
    encoder_input_ids: List[int]
    decoder_input_ids: List[int]  # decoder_target_ids shifted right
    labels: List[int]  # Same as decoder_target_ids
    task_type: UL2TaskType
    
    # Original for debugging
    original_ids: Optional[List[int]] = None


class UL2Corruptor:
    """
    Main class for applying UL2 corruption to text.
    
    Handles task sampling and corruption application.
    """
    
    def __init__(
        self,
        tokenizer,
        tasks: List[UL2TaskConfig] = None,
        max_sentinel_tokens: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize UL2 corruptor.
        
        Args:
            tokenizer: Qwen3EncoderDecoderTokenizer with sentinel tokens
            tasks: UL2 task configurations
            max_sentinel_tokens: Maximum number of sentinel tokens available
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.task_sampler = UL2TaskSampler(tasks)
        self.max_sentinels = max_sentinel_tokens
        
        self.random_state = np.random.RandomState(seed)
    
    def _get_sentinel_id(self, sentinel_idx: int) -> int:
        """Get sentinel token ID."""
        if sentinel_idx >= self.max_sentinels:
            # Wrap around if we run out (rare for most corruption rates)
            sentinel_idx = sentinel_idx % self.max_sentinels
        return self.tokenizer.get_sentinel_token_id(sentinel_idx)
    
    def corrupt(
        self,
        token_ids: List[int],
        task: Optional[UL2TaskConfig] = None,
    ) -> UL2CorruptedExample:
        """
        Apply UL2 corruption to token sequence.
        
        Args:
            token_ids: Original token IDs
            task: Specific task to apply (samples if None)
            
        Returns:
            UL2CorruptedExample with encoder/decoder sequences
        """
        if task is None:
            task = self.task_sampler.sample()
        
        sequence_length = len(token_ids)
        
        if task.task_type == UL2TaskType.S_DENOISER:
            # Sequential denoising
            encoder_ids, decoder_target_ids = apply_s_denoising(
                token_ids,
                task.length_fraction,
                self._get_sentinel_id,
                self.random_state,
            )
        else:
            # Span corruption (R or X denoiser)
            span_boundaries = compute_span_boundaries(
                sequence_length,
                task.mean_span_length,
                task.corruption_rate,
                self.random_state,
            )
            
            encoder_ids, decoder_target_ids = apply_span_corruption(
                token_ids,
                span_boundaries,
                self._get_sentinel_id,
            )
        
        # Create decoder input (shifted right)
        # Prepend decoder_start_token_id (usually pad or special token)
        decoder_start_id = self.tokenizer.pad_token_id  # Or dedicated start token
        decoder_input_ids = [decoder_start_id] + decoder_target_ids[:-1]
        
        return UL2CorruptedExample(
            encoder_input_ids=encoder_ids,
            decoder_input_ids=decoder_input_ids,
            labels=decoder_target_ids,
            task_type=task.task_type,
            original_ids=token_ids,
        )
    
    def corrupt_text(
        self,
        text: str,
        task: Optional[UL2TaskConfig] = None,
    ) -> UL2CorruptedExample:
        """
        Tokenize and corrupt text.
        
        Args:
            text: Input text
            task: Specific task (samples if None)
            
        Returns:
            UL2CorruptedExample
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.corrupt(token_ids, task)
    
    def corrupt_batch(
        self,
        batch_token_ids: List[List[int]],
        tasks: Optional[List[UL2TaskConfig]] = None,
    ) -> List[UL2CorruptedExample]:
        """
        Corrupt a batch of sequences.
        
        Args:
            batch_token_ids: List of token ID sequences
            tasks: List of tasks (samples if None)
            
        Returns:
            List of UL2CorruptedExample
        """
        if tasks is None:
            tasks = self.task_sampler.sample_batch(len(batch_token_ids))
        
        return [
            self.corrupt(token_ids, task)
            for token_ids, task in zip(batch_token_ids, tasks)
        ]
```

### 7.2 Data Collator

```python
# src/data/collator.py

"""
Data collator for UL2 encoder-decoder training.

Handles padding, attention masks, and label preparation.
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizerBase

from .ul2_corruption import UL2CorruptedExample


@dataclass
class UL2DataCollator:
    """
    Data collator for UL2 training.
    
    Pads encoder and decoder sequences and creates attention masks.
    """
    
    tokenizer: PreTrainedTokenizerBase
    max_encoder_length: int = 2048
    max_decoder_length: int = 1024
    pad_to_multiple_of: Optional[int] = 8  # For tensor core efficiency
    label_pad_token_id: int = -100  # Ignored in loss computation
    
    def __call__(
        self,
        examples: List[Union[UL2CorruptedExample, Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of UL2CorruptedExample or dicts with keys:
                - encoder_input_ids
                - decoder_input_ids
                - labels
                
        Returns:
            Dictionary with batched tensors:
                - input_ids: Encoder input [batch, enc_len]
                - attention_mask: Encoder attention mask [batch, enc_len]
                - decoder_input_ids: Decoder input [batch, dec_len]
                - decoder_attention_mask: Decoder attention mask [batch, dec_len]
                - labels: Target labels [batch, dec_len]
        """
        # Extract sequences
        if isinstance(examples[0], UL2CorruptedExample):
            encoder_ids = [ex.encoder_input_ids for ex in examples]
            decoder_ids = [ex.decoder_input_ids for ex in examples]
            labels = [ex.labels for ex in examples]
        else:
            encoder_ids = [ex["encoder_input_ids"] for ex in examples]
            decoder_ids = [ex["decoder_input_ids"] for ex in examples]
            labels = [ex["labels"] for ex in examples]
        
        # Truncate if necessary
        encoder_ids = [ids[:self.max_encoder_length] for ids in encoder_ids]
        decoder_ids = [ids[:self.max_decoder_length] for ids in decoder_ids]
        labels = [lbl[:self.max_decoder_length] for lbl in labels]
        
        # Compute max lengths
        max_enc_len = max(len(ids) for ids in encoder_ids)
        max_dec_len = max(len(ids) for ids in decoder_ids)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_enc_len = (
                (max_enc_len + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
            max_dec_len = (
                (max_dec_len + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
        
        batch_size = len(examples)
        pad_id = self.tokenizer.pad_token_id
        
        # Initialize tensors
        input_ids = torch.full(
            (batch_size, max_enc_len), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_enc_len), dtype=torch.long
        )
        decoder_input_ids = torch.full(
            (batch_size, max_dec_len), pad_id, dtype=torch.long
        )
        decoder_attention_mask = torch.zeros(
            (batch_size, max_dec_len), dtype=torch.long
        )
        label_tensor = torch.full(
            (batch_size, max_dec_len), self.label_pad_token_id, dtype=torch.long
        )
        
        # Fill tensors
        for i, (enc_ids, dec_ids, lbl) in enumerate(zip(encoder_ids, decoder_ids, labels)):
            enc_len = len(enc_ids)
            dec_len = len(dec_ids)
            
            input_ids[i, :enc_len] = torch.tensor(enc_ids)
            attention_mask[i, :enc_len] = 1
            
            decoder_input_ids[i, :dec_len] = torch.tensor(dec_ids)
            decoder_attention_mask[i, :dec_len] = 1
            
            label_tensor[i, :len(lbl)] = torch.tensor(lbl)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": label_tensor,
        }


@dataclass  
class UL2DataCollatorWithTaskPrefixes(UL2DataCollator):
    """
    Data collator that optionally prepends task prefixes.
    
    T5Gemma 2 uses task prefixes like "[NLU]", "[NLG]", "[S2S]"
    to help the model identify the denoising task.
    """
    
    add_task_prefixes: bool = True
    
    # Task prefix tokens (should be added to tokenizer)
    task_prefix_map: Dict[str, str] = None
    
    def __post_init__(self):
        if self.task_prefix_map is None:
            self.task_prefix_map = {
                "r1": "[R]",
                "r2": "[R]", 
                "x1": "[X]",
                "x2": "[X]",
                "s": "[S]",
            }
    
    def _add_task_prefix(
        self,
        encoder_ids: List[int],
        task_type: str,
    ) -> List[int]:
        """Add task prefix to encoder input."""
        if not self.add_task_prefixes:
            return encoder_ids
        
        prefix_text = self.task_prefix_map.get(task_type, "")
        if prefix_text:
            prefix_ids = self.tokenizer.encode(
                prefix_text, add_special_tokens=False
            )
            return prefix_ids + encoder_ids
        
        return encoder_ids
    
    def __call__(
        self,
        examples: List[Union[UL2CorruptedExample, Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """Collate with optional task prefixes."""
        if self.add_task_prefixes:
            processed = []
            for ex in examples:
                if isinstance(ex, UL2CorruptedExample):
                    task_type = ex.task_type.value
                    encoder_ids = self._add_task_prefix(
                        ex.encoder_input_ids, task_type
                    )
                    processed.append({
                        "encoder_input_ids": encoder_ids,
                        "decoder_input_ids": ex.decoder_input_ids,
                        "labels": ex.labels,
                    })
                else:
                    processed.append(ex)
            examples = processed
        
        return super().__call__(examples)
```

### 7.3 Streaming Dataset Integration

```python
# src/data/ul2_dataset.py

"""
UL2 dataset with HuggingFace Datasets streaming integration.

Supports large-scale training on trillion-token datasets.
"""

import logging
from typing import Optional, Iterator, Dict, Any, List, Union
from functools import partial

from datasets import load_dataset, IterableDataset, Dataset
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from .ul2_corruption import UL2Corruptor, UL2CorruptedExample, UL2TaskConfig
from ..tokenization_qwen3_encdec import Qwen3EncoderDecoderTokenizer

logger = logging.getLogger(__name__)


class UL2StreamingDataset(TorchIterableDataset):
    """
    Streaming dataset for UL2 training.
    
    Wraps HuggingFace IterableDataset with on-the-fly UL2 corruption.
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        tokenizer: Qwen3EncoderDecoderTokenizer,
        corruptor: UL2Corruptor,
        text_column: str = "text",
        max_seq_length: int = 8192,
        buffer_size: int = 10000,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            dataset: HuggingFace IterableDataset
            tokenizer: Tokenizer with sentinel tokens
            corruptor: UL2 corruptor instance
            text_column: Column name containing text
            max_seq_length: Maximum sequence length before corruption
            buffer_size: Shuffle buffer size
        """
        self.dataset = dataset.shuffle(buffer_size=buffer_size)
        self.tokenizer = tokenizer
        self.corruptor = corruptor
        self.text_column = text_column
        self.max_seq_length = max_seq_length
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over corrupted examples."""
        for example in self.dataset:
            text = example[self.text_column]
            
            # Tokenize
            token_ids = self.tokenizer.encode(
                text, 
                add_special_tokens=False,
                max_length=self.max_seq_length,
                truncation=True,
            )
            
            # Skip very short sequences
            if len(token_ids) < 10:
                continue
            
            # Apply corruption
            corrupted = self.corruptor.corrupt(token_ids)
            
            yield {
                "encoder_input_ids": corrupted.encoder_input_ids,
                "decoder_input_ids": corrupted.decoder_input_ids,
                "labels": corrupted.labels,
                "task_type": corrupted.task_type.value,
            }


def create_ul2_dataset(
    dataset_name: str,
    tokenizer: Qwen3EncoderDecoderTokenizer,
    split: str = "train",
    streaming: bool = True,
    text_column: str = "text",
    max_seq_length: int = 8192,
    ul2_tasks: Optional[List[UL2TaskConfig]] = None,
    seed: int = 42,
    **dataset_kwargs,
) -> Union[UL2StreamingDataset, Dataset]:
    """
    Create UL2 dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb-edu")
        tokenizer: Tokenizer with sentinel tokens
        split: Dataset split
        streaming: Whether to use streaming mode
        text_column: Column containing text
        max_seq_length: Maximum sequence length
        ul2_tasks: Custom UL2 task configurations
        seed: Random seed
        **dataset_kwargs: Additional arguments for load_dataset
        
    Returns:
        UL2 dataset ready for training
        
    Example:
        >>> tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained("...")
        >>> dataset = create_ul2_dataset(
        ...     "HuggingFaceFW/fineweb-edu",
        ...     tokenizer,
        ...     streaming=True,
        ... )
    """
    logger.info(f"Loading dataset {dataset_name} (streaming={streaming})")
    
    # Load dataset
    raw_dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        **dataset_kwargs,
    )
    
    # Create corruptor
    corruptor = UL2Corruptor(
        tokenizer=tokenizer,
        tasks=ul2_tasks,
        seed=seed,
    )
    
    if streaming:
        return UL2StreamingDataset(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            corruptor=corruptor,
            text_column=text_column,
            max_seq_length=max_seq_length,
        )
    else:
        # Map corruption function for non-streaming
        def corrupt_example(example):
            text = example[text_column]
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_seq_length,
                truncation=True,
            )
            
            if len(token_ids) < 10:
                return {
                    "encoder_input_ids": [],
                    "decoder_input_ids": [],
                    "labels": [],
                    "task_type": "",
                }
            
            corrupted = corruptor.corrupt(token_ids)
            return {
                "encoder_input_ids": corrupted.encoder_input_ids,
                "decoder_input_ids": corrupted.decoder_input_ids,
                "labels": corrupted.labels,
                "task_type": corrupted.task_type.value,
            }
        
        return raw_dataset.map(
            corrupt_example,
            remove_columns=raw_dataset.column_names,
            desc="Applying UL2 corruption",
        ).filter(lambda x: len(x["encoder_input_ids"]) > 0)


class MultiDatasetMixer:
    """
    Mix multiple datasets for training.
    
    Useful for combining different data sources with different weights.
    """
    
    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ):
        """
        Initialize mixer.
        
        Args:
            datasets: List of datasets to mix
            weights: Sampling weights (uniform if None)
            seed: Random seed
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        self.rng = torch.Generator().manual_seed(seed)
    
    def __iter__(self):
        """Iterate over mixed datasets."""
        iterators = [iter(ds) for ds in self.datasets]
        
        while True:
            # Sample dataset according to weights
            idx = torch.multinomial(
                torch.tensor(self.weights),
                1,
                generator=self.rng,
            ).item()
            
            try:
                yield next(iterators[idx])
            except StopIteration:
                # Restart exhausted iterator
                iterators[idx] = iter(self.datasets[idx])
                yield next(iterators[idx])
```

### 7.4 Data Pipeline Script

```python
# scripts/prepare_ul2_data.py

"""
Script to prepare and test UL2 data pipeline.

Usage:
    python scripts/prepare_ul2_data.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --tokenizer ./qwen3-encdec-tokenizer \
        --num-examples 1000 \
        --output ./ul2_sample_data
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter

from datasets import load_dataset

from src.tokenization_qwen3_encdec import Qwen3EncoderDecoderTokenizer
from src.data.ul2_corruption import UL2Corruptor, DEFAULT_UL2_TASKS
from src.data.ul2_dataset import create_ul2_dataset
from src.data.collator import UL2DataCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_corruption_statistics(
    tokenizer,
    num_samples: int = 1000,
    text_samples: list = None,
):
    """Analyze UL2 corruption statistics."""
    corruptor = UL2Corruptor(tokenizer, seed=42)
    
    stats = {
        "task_distribution": Counter(),
        "encoder_lengths": [],
        "decoder_lengths": [],
        "compression_ratios": [],
    }
    
    # Use sample texts if provided, else generate dummy
    if text_samples is None:
        text_samples = [
            "This is a sample text for testing UL2 corruption. " * 10
        ] * num_samples
    
    for text in text_samples[:num_samples]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 10:
            continue
            
        corrupted = corruptor.corrupt(token_ids)
        
        stats["task_distribution"][corrupted.task_type.value] += 1
        stats["encoder_lengths"].append(len(corrupted.encoder_input_ids))
        stats["decoder_lengths"].append(len(corrupted.decoder_input_ids))
        stats["compression_ratios"].append(
            len(corrupted.encoder_input_ids) / len(token_ids)
        )
    
    # Compute summary statistics
    import numpy as np
    
    summary = {
        "task_distribution": dict(stats["task_distribution"]),
        "encoder_length": {
            "mean": np.mean(stats["encoder_lengths"]),
            "std": np.std(stats["encoder_lengths"]),
            "min": np.min(stats["encoder_lengths"]),
            "max": np.max(stats["encoder_lengths"]),
        },
        "decoder_length": {
            "mean": np.mean(stats["decoder_lengths"]),
            "std": np.std(stats["decoder_lengths"]),
            "min": np.min(stats["decoder_lengths"]),
            "max": np.max(stats["decoder_lengths"]),
        },
        "compression_ratio": {
            "mean": np.mean(stats["compression_ratios"]),
            "std": np.std(stats["compression_ratios"]),
        },
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="./ul2_sample_data")
    parser.add_argument("--analyze-only", action="store_true")
    
    args = parser.parse_args()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(args.tokenizer)
    
    if args.analyze_only:
        # Just analyze corruption statistics
        logger.info("Analyzing UL2 corruption statistics...")
        stats = analyze_corruption_statistics(tokenizer, args.num_examples)
        
        print("\n=== UL2 Corruption Statistics ===")
        print(f"Task distribution: {stats['task_distribution']}")
        print(f"Encoder length: {stats['encoder_length']}")
        print(f"Decoder length: {stats['decoder_length']}")
        print(f"Compression ratio: {stats['compression_ratio']}")
        return
    
    # Create dataset
    logger.info(f"Creating UL2 dataset from {args.dataset}")
    dataset = create_ul2_dataset(
        args.dataset,
        tokenizer,
        streaming=True,
        max_seq_length=2048,
    )
    
    # Sample examples
    logger.info(f"Sampling {args.num_examples} examples...")
    examples = []
    for i, example in enumerate(dataset):
        if i >= args.num_examples:
            break
        examples.append(example)
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1} examples")
    
    # Test collator
    logger.info("Testing data collator...")
    collator = UL2DataCollator(tokenizer, max_encoder_length=512, max_decoder_length=256)
    batch = collator(examples[:8])
    
    print("\n=== Sample Batch Shapes ===")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape}")
    
    # Save samples
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "samples.json", "w") as f:
        # Convert to serializable format
        serializable = []
        for ex in examples[:100]:
            serializable.append({
                "encoder_input_ids": ex["encoder_input_ids"],
                "decoder_input_ids": ex["decoder_input_ids"],
                "labels": ex["labels"],
                "task_type": ex["task_type"],
            })
        json.dump(serializable, f, indent=2)
    
    logger.info(f"Saved {len(serializable)} samples to {output_path / 'samples.json'}")


if __name__ == "__main__":
    main()
```

---

## Unit Tests

```python
# tests/test_ul2_pipeline.py

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.data.ul2_corruption import (
    UL2TaskType,
    UL2TaskConfig,
    UL2TaskSampler,
    compute_span_boundaries,
    apply_span_corruption,
    apply_s_denoising,
    UL2Corruptor,
    DEFAULT_UL2_TASKS,
)
from src.data.collator import UL2DataCollator


class TestUL2TaskSampler:
    """Tests for task sampling."""
    
    def test_default_weights(self):
        """Test default task weights (1:1:1:1:4)."""
        sampler = UL2TaskSampler()
        
        # Sample many times and check distribution
        samples = [sampler.sample() for _ in range(8000)]
        
        task_counts = {}
        for s in samples:
            task_counts[s.task_type] = task_counts.get(s.task_type, 0) + 1
        
        # S-denoiser should be ~4x more common
        s_count = task_counts.get(UL2TaskType.S_DENOISER, 0)
        r1_count = task_counts.get(UL2TaskType.R_DENOISER_1, 0)
        
        # Allow 20% deviation due to randomness
        ratio = s_count / max(r1_count, 1)
        assert 3.0 < ratio < 5.0, f"S/R1 ratio {ratio} outside expected range"
    
    def test_batch_sampling(self):
        """Test batch sampling returns correct count."""
        sampler = UL2TaskSampler()
        
        batch = sampler.sample_batch(100)
        
        assert len(batch) == 100
        assert all(isinstance(t, UL2TaskConfig) for t in batch)


class TestSpanCorruption:
    """Tests for span corruption logic."""
    
    def test_compute_span_boundaries_basic(self):
        """Test basic span boundary computation."""
        boundaries = compute_span_boundaries(
            sequence_length=100,
            mean_span_length=3,
            corruption_rate=0.15,
            random_state=np.random.RandomState(42),
        )
        
        # Should have some spans
        assert len(boundaries) > 0
        
        # Spans should not overlap
        sorted_spans = sorted(boundaries, key=lambda x: x[0])
        for i in range(len(sorted_spans) - 1):
            assert sorted_spans[i][1] <= sorted_spans[i + 1][0]
    
    def test_compute_span_boundaries_respects_rate(self):
        """Test that corruption rate is approximately respected."""
        random_state = np.random.RandomState(42)
        total_corrupted = 0
        total_length = 0
        
        for _ in range(100):
            seq_len = 1000
            boundaries = compute_span_boundaries(
                sequence_length=seq_len,
                mean_span_length=10,
                corruption_rate=0.3,
                random_state=random_state,
            )
            
            corrupted = sum(end - start for start, end in boundaries)
            total_corrupted += corrupted
            total_length += seq_len
        
        actual_rate = total_corrupted / total_length
        # Allow 20% deviation
        assert 0.24 < actual_rate < 0.36, f"Actual rate {actual_rate} outside expected range"
    
    def test_apply_span_corruption_basic(self):
        """Test basic span corruption application."""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        span_boundaries = [(1, 3), (5, 7)]  # Corrupt tokens at indices 1,2 and 5,6
        
        get_sentinel = lambda idx: 100 + idx  # Sentinel IDs: 100, 101, ...
        
        encoder_ids, decoder_ids = apply_span_corruption(
            token_ids, span_boundaries, get_sentinel
        )
        
        # Encoder should have: [1, <s0>, 4, 5, <s1>, 8]
        assert encoder_ids == [1, 100, 4, 5, 101, 8]
        
        # Decoder should have: [<s0>, 2, 3, <s1>, 6, 7]
        assert decoder_ids == [100, 2, 3, 101, 6, 7]
    
    def test_apply_span_corruption_preserves_all_tokens(self):
        """Test that all original tokens appear in encoder + decoder."""
        token_ids = list(range(100))
        span_boundaries = [(10, 20), (50, 60), (80, 90)]
        
        get_sentinel = lambda idx: 1000 + idx
        
        encoder_ids, decoder_ids = apply_span_corruption(
            token_ids, span_boundaries, get_sentinel
        )
        
        # Extract non-sentinel tokens
        encoder_tokens = [t for t in encoder_ids if t < 1000]
        decoder_tokens = [t for t in decoder_ids if t < 1000]
        
        # All original tokens should appear exactly once
        all_tokens = encoder_tokens + decoder_tokens
        assert sorted(all_tokens) == token_ids
    
    def test_apply_span_corruption_empty(self):
        """Test with no spans to corrupt."""
        token_ids = [1, 2, 3, 4, 5]
        
        encoder_ids, decoder_ids = apply_span_corruption(
            token_ids, [], lambda x: 100 + x
        )
        
        assert encoder_ids == token_ids
        assert decoder_ids == []


class TestSDenoising:
    """Tests for S-Denoiser (sequential) corruption."""
    
    def test_s_denoising_basic(self):
        """Test basic S-denoising."""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        
        encoder_ids, decoder_ids = apply_s_denoising(
            token_ids,
            length_fraction=0.75,  # 6 tokens as suffix
            get_sentinel_id_fn=lambda x: 100,
        )
        
        # Encoder should be prefix + sentinel
        # Decoder should be sentinel + suffix
        assert len(encoder_ids) == 3  # 2 prefix + 1 sentinel
        assert encoder_ids[-1] == 100  # Ends with sentinel
        
        assert decoder_ids[0] == 100  # Starts with sentinel
        assert len(decoder_ids) == 7  # 1 sentinel + 6 suffix
    
    def test_s_denoising_preserves_all_tokens(self):
        """Test that all tokens appear in result."""
        token_ids = list(range(100))
        
        encoder_ids, decoder_ids = apply_s_denoising(
            token_ids,
            length_fraction=0.5,
            get_sentinel_id_fn=lambda x: 1000,
        )
        
        # Extract non-sentinel tokens
        encoder_tokens = [t for t in encoder_ids if t < 1000]
        decoder_tokens = [t for t in decoder_ids if t < 1000]
        
        # All original tokens should appear
        all_tokens = encoder_tokens + decoder_tokens
        assert sorted(all_tokens) == token_ids


class TestUL2Corruptor:
    """Tests for UL2Corruptor class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=list(range(100)))
        tokenizer.pad_token_id = 0
        tokenizer.get_sentinel_token_id = Mock(side_effect=lambda x: 1000 + x)
        return tokenizer
    
    def test_corrupt_returns_valid_structure(self, mock_tokenizer):
        """Test that corrupt returns valid UL2CorruptedExample."""
        corruptor = UL2Corruptor(mock_tokenizer, seed=42)
        
        token_ids = list(range(100))
        result = corruptor.corrupt(token_ids)
        
        assert hasattr(result, 'encoder_input_ids')
        assert hasattr(result, 'decoder_input_ids')
        assert hasattr(result, 'labels')
        assert hasattr(result, 'task_type')
        
        assert isinstance(result.encoder_input_ids, list)
        assert isinstance(result.task_type, UL2TaskType)
    
    def test_corrupt_text(self, mock_tokenizer):
        """Test corrupt_text method."""
        corruptor = UL2Corruptor(mock_tokenizer, seed=42)
        
        result = corruptor.corrupt_text("Sample text for testing")
        
        mock_tokenizer.encode.assert_called()
        assert result.encoder_input_ids is not None
    
    def test_corrupt_batch(self, mock_tokenizer):
        """Test batch corruption."""
        corruptor = UL2Corruptor(mock_tokenizer, seed=42)
        
        batch = [list(range(50)), list(range(100)), list(range(75))]
        results = corruptor.corrupt_batch(batch)
        
        assert len(results) == 3
        assert all(isinstance(r.task_type, UL2TaskType) for r in results)


class TestUL2DataCollator:
    """Tests for data collator."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        return tokenizer
    
    def test_collator_basic(self, mock_tokenizer):
        """Test basic collation."""
        collator = UL2DataCollator(
            tokenizer=mock_tokenizer,
            max_encoder_length=100,
            max_decoder_length=50,
        )
        
        examples = [
            {
                "encoder_input_ids": [1, 2, 3, 4, 5],
                "decoder_input_ids": [10, 11, 12],
                "labels": [11, 12, 13],
            },
            {
                "encoder_input_ids": [1, 2, 3],
                "decoder_input_ids": [10, 11],
                "labels": [11, 12],
            },
        ]
        
        batch = collator(examples)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "decoder_input_ids" in batch
        assert "labels" in batch
        
        assert batch["input_ids"].shape[0] == 2
        assert batch["decoder_input_ids"].shape[0] == 2
    
    def test_collator_padding(self, mock_tokenizer):
        """Test that padding is applied correctly."""
        collator = UL2DataCollator(
            tokenizer=mock_tokenizer,
            pad_to_multiple_of=None,  # Disable for exact testing
        )
        
        examples = [
            {
                "encoder_input_ids": [1, 2, 3, 4, 5],
                "decoder_input_ids": [10, 11, 12],
                "labels": [11, 12, 13],
            },
            {
                "encoder_input_ids": [1, 2],
                "decoder_input_ids": [10],
                "labels": [11],
            },
        ]
        
        batch = collator(examples)
        
        # Should be padded to max length in batch
        assert batch["input_ids"].shape[1] == 5
        assert batch["decoder_input_ids"].shape[1] == 3
        
        # Check padding values
        assert batch["input_ids"][1, 2].item() == 0  # Pad token
        assert batch["labels"][1, 1].item() == -100  # Label pad
    
    def test_collator_attention_mask(self, mock_tokenizer):
        """Test attention mask correctness."""
        collator = UL2DataCollator(
            tokenizer=mock_tokenizer,
            pad_to_multiple_of=None,
        )
        
        examples = [
            {
                "encoder_input_ids": [1, 2, 3],
                "decoder_input_ids": [10, 11],
                "labels": [11, 12],
            },
        ]
        
        batch = collator(examples)
        
        # All positions should be attended (no padding in single example)
        assert batch["attention_mask"][0].sum().item() == 3
        assert batch["decoder_attention_mask"][0].sum().item() == 2
    
    def test_collator_truncation(self, mock_tokenizer):
        """Test that sequences are truncated to max length."""
        collator = UL2DataCollator(
            tokenizer=mock_tokenizer,
            max_encoder_length=5,
            max_decoder_length=3,
            pad_to_multiple_of=None,
        )
        
        examples = [
            {
                "encoder_input_ids": list(range(100)),
                "decoder_input_ids": list(range(50)),
                "labels": list(range(50)),
            },
        ]
        
        batch = collator(examples)
        
        assert batch["input_ids"].shape[1] == 5
        assert batch["decoder_input_ids"].shape[1] == 3


class TestCorruptionStatistics:
    """Statistical tests for corruption behavior."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=list(range(1000)))
        tokenizer.pad_token_id = 0
        tokenizer.get_sentinel_token_id = Mock(side_effect=lambda x: 10000 + x)
        return tokenizer
    
    def test_r_denoiser_short_spans(self, mock_tokenizer):
        """Test R-denoiser produces short spans."""
        task = UL2TaskConfig(
            task_type=UL2TaskType.R_DENOISER_1,
            mean_span_length=3,
            corruption_rate=0.15,
        )
        
        corruptor = UL2Corruptor(mock_tokenizer, tasks=[task], seed=42)
        
        span_lengths = []
        for _ in range(100):
            token_ids = list(range(500))
            result = corruptor.corrupt(token_ids)
            
            # Count consecutive corrupted tokens (sentinel markers)
            # Spans appear in decoder
            span_lengths.append(len(result.labels))
        
        mean_decoder_len = np.mean(span_lengths)
        # Decoder should be relatively short for R-denoiser
        assert mean_decoder_len < 200  # ~15% of 500 + sentinels
    
    def test_x_denoiser_long_spans(self, mock_tokenizer):
        """Test X-denoiser produces longer spans."""
        task = UL2TaskConfig(
            task_type=UL2TaskType.X_DENOISER_2,
            mean_span_length=32,
            corruption_rate=0.50,
        )
        
        corruptor = UL2Corruptor(mock_tokenizer, tasks=[task], seed=42)
        
        decoder_lengths = []
        for _ in range(100):
            token_ids = list(range(500))
            result = corruptor.corrupt(token_ids)
            decoder_lengths.append(len(result.labels))
        
        mean_decoder_len = np.mean(decoder_lengths)
        # Decoder should be about 50% of input
        assert 200 < mean_decoder_len < 350
```

---

## Acceptance Criteria

1. **Task Implementation**
   - [ ] All 5 UL2 tasks implemented (R1, R2, X1, X2, S)
   - [ ] Task sampling follows 1:1:1:1:4 weighting
   - [ ] Span corruption uses geometric distribution for lengths
   - [ ] S-denoiser correctly splits prefix/suffix

2. **Corruption Quality**
   - [ ] All original tokens preserved in encoder + decoder
   - [ ] Sentinel tokens correctly placed
   - [ ] Corruption rates approximately match configuration
   - [ ] No overlapping spans

3. **Data Pipeline**
   - [ ] Streaming dataset works with HuggingFace datasets
   - [ ] Data collator produces correct tensor shapes
   - [ ] Attention masks correctly set
   - [ ] Labels padded with -100 for loss masking

4. **Performance**
   - [ ] Streaming mode doesn't load full dataset into memory
   - [ ] Corruption is efficient (< 1ms per example)
   - [ ] Collator handles variable-length batches

5. **Testing**
   - [ ] Unit tests cover all corruption scenarios
   - [ ] Statistical tests verify distribution properties
   - [ ] Integration test with real tokenizer

---

## Dependencies

- **Story 02**: Tokenizer with sentinel tokens

---

## Estimated Effort

- UL2 corruption implementation: 1.5 days
- Data collator: 0.5 days
- Streaming dataset integration: 1 day
- Testing and validation: 1 day
- **Total: 3-4 days**

---

## Developer Notes

1. **Span Length Distribution**: T5 uses geometric distribution for span lengths. The parameter p = 1/mean_span_length gives the desired mean.

2. **Corruption Rate vs Span Count**: Higher corruption rate with longer mean spans = fewer but longer spans. Lower rate with short spans = many small spans.

3. **S-Denoiser Importance**: The 4× weight for S-denoiser is crucial - it teaches the model causal generation which transfers to downstream tasks.

4. **Memory Efficiency**: Use streaming datasets for large-scale training. The corruption is applied on-the-fly, so only one batch is in memory at a time.

5. **Debugging Tips**:
   - Print corrupted examples to verify sentinel placement
   - Check that task distribution matches expected weights
   - Verify encoder + decoder tokens = original tokens + sentinels

6. **Reference Implementation**: Study T5's data pipeline in the original paper and HuggingFace's T5 implementation.
