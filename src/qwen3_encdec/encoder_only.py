"""Standalone encoder model for embedding tasks.

This module provides an encoder-only model extracted from the trained
Qwen3 encoder-decoder, optimized for embedding and retrieval tasks.

Features:
- Multiple pooling strategies (mean, cls, last, weighted_mean)
- L2 normalization option
- Batch encoding with progress bar
- Sentence-transformers compatibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_encoder import Qwen3Encoder

if TYPE_CHECKING:
    from .modeling_qwen3_encdec import Qwen3ForSeq2SeqLM


# =============================================================================
# Configuration
# =============================================================================


class Qwen3EncoderConfig(PretrainedConfig):
    """
    Configuration for standalone Qwen3 Encoder model.

    This is a simplified config for the encoder-only model,
    derived from the full Qwen3EncoderDecoderConfig.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Size of hidden layers.
        intermediate_size: Size of MLP intermediate layers.
        num_hidden_layers: Number of encoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads (for GQA).
        head_dim: Dimension of each attention head.
        hidden_act: Activation function.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base for rotary position embeddings.
        attention_dropout: Dropout rate for attention.
        pad_token_id: Padding token ID.
        pooling_mode: Pooling strategy (mean, cls, last, weighted_mean).
        normalize_embeddings: Whether to L2-normalize output embeddings.
    """

    model_type = "qwen3_encoder"

    def __init__(
        self,
        vocab_size: int = 152036,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,  # Qwen3-0.6B uses 128
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        pad_token_id: int = 151643,
        pooling_mode: str = "mean",
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        # Embedding-specific settings
        self.pooling_mode = pooling_mode
        self.normalize_embeddings = normalize_embeddings

    @classmethod
    def from_encoder_decoder_config(
        cls,
        config: Qwen3EncoderDecoderConfig,
        pooling_mode: str = "mean",
        normalize_embeddings: bool = True,
    ) -> "Qwen3EncoderConfig":
        """Create encoder config from full encoder-decoder config."""
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            pad_token_id=config.pad_token_id,
            pooling_mode=pooling_mode,
            normalize_embeddings=normalize_embeddings,
        )


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
class Qwen3EncoderPoolerOutput(BaseModelOutput):
    """
    Output type for the encoder model with pooling.

    Args:
        last_hidden_state: Sequence of hidden-states at the output of the last layer.
            Shape: (batch_size, sequence_length, hidden_size)
        pooler_output: Pooled representation of the sequence.
            Shape: (batch_size, hidden_size)
        hidden_states: Tuple of hidden-states at each layer (if output_hidden_states=True).
        attentions: Tuple of attention weights at each layer (if output_attentions=True).
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# =============================================================================
# Pooler
# =============================================================================


class Qwen3EncoderPooler(nn.Module):
    """
    Pooling strategies for encoder outputs.

    Supports:
    - mean: Mean pooling over non-padding tokens
    - cls: First token (CLS-style)
    - last: Last non-padding token
    - weighted_mean: Position-weighted mean pooling (later positions weighted more)

    Args:
        config: Encoder configuration with pooling settings.
    """

    def __init__(self, config: Qwen3EncoderConfig):
        super().__init__()
        self.pooling_mode = config.pooling_mode
        self.normalize = config.normalize_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool hidden states into a single vector per sequence.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), 1 for valid tokens, 0 for padding

        Returns:
            Pooled output: (batch_size, hidden_size)
        """
        if self.pooling_mode == "cls":
            pooled = hidden_states[:, 0]

        elif self.pooling_mode == "last":
            # Get last non-padding token for each sequence
            batch_size = hidden_states.size(0)
            # Find the last valid position for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            sequence_lengths = sequence_lengths.clamp(min=0)  # Handle empty sequences
            pooled = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths.long(),
            ]

        elif self.pooling_mode == "mean":
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask

        elif self.pooling_mode == "weighted_mean":
            # Position-weighted mean (later positions weighted more)
            batch_size, seq_len, hidden_size = hidden_states.size()

            # Create position weights (linear increase)
            positions = torch.arange(seq_len, device=hidden_states.device).float()
            weights = (positions + 1).unsqueeze(0).expand(batch_size, -1)  # (batch, seq)

            # Apply attention mask
            weights = weights * attention_mask.float()

            # Normalize weights
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)

            # Weighted sum
            pooled = torch.sum(
                hidden_states * weights.unsqueeze(-1),
                dim=1,
            )

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        # Normalize if requested
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled


# =============================================================================
# Standalone Encoder Model
# =============================================================================


class Qwen3StandaloneEncoderModel(PreTrainedModel):
    """
    Standalone Qwen3 Encoder for embedding tasks.

    This model wraps the encoder from the trained encoder-decoder
    and adds pooling for sentence/document embeddings.

    Example:
        ```python
        from transformers import AutoTokenizer
        from qwen3_encdec import Qwen3StandaloneEncoderModel

        model = Qwen3StandaloneEncoderModel.from_pretrained("path/to/encoder")
        tokenizer = AutoTokenizer.from_pretrained("path/to/encoder")

        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model(**inputs)

        # Get sentence embedding
        embedding = outputs.pooler_output  # (1, hidden_size)
        ```
    """

    config_class = Qwen3EncoderConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3EncoderConfig):
        super().__init__(config)

        # Create encoder-decoder config for encoder initialization
        # (Qwen3Encoder expects Qwen3EncoderDecoderConfig)
        # We need to set sentinel tokens correctly for validation
        sentinel_start = config.vocab_size - 100 if config.vocab_size > 100 else 0
        num_sentinels = config.vocab_size - sentinel_start if config.vocab_size > 100 else 0

        self._enc_dec_config = Qwen3EncoderDecoderConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            pad_token_id=config.pad_token_id,
            sentinel_token_start_id=sentinel_start,
            num_sentinel_tokens=num_sentinels,
        )

        # Encoder layers
        self.encoder = Qwen3Encoder(self._enc_dec_config)

        # Pooler
        self.pooler = Qwen3EncoderPooler(config)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embeddings."""
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input embeddings."""
        self.encoder.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen3EncoderPoolerOutput]:
        """
        Forward pass for encoding.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Optional position indices
            inputs_embeds: Optional pre-computed embeddings
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dataclass

        Returns:
            Qwen3EncoderPoolerOutput with last_hidden_state and pooler_output
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create attention mask if not provided
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
            elif inputs_embeds is not None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )

        # Run encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state

        # Pool to get sentence embedding
        pooler_output = self.pooler(last_hidden_state, attention_mask)

        if not return_dict:
            return (last_hidden_state, pooler_output) + (
                encoder_outputs.hidden_states,
                encoder_outputs.attentions,
            )

        return Qwen3EncoderPoolerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def encode(
        self,
        sentences: Union[str, List[str]],
        tokenizer,
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = True,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode sentences to embeddings.

        Convenience method for encoding multiple sentences.

        Args:
            sentences: Single sentence or list of sentences
            tokenizer: Tokenizer to use
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar
            device: Device to use (default: model's device)

        Returns:
            Embeddings tensor (num_sentences, hidden_size)
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]

        device = device or next(self.parameters()).device
        self.eval()

        all_embeddings = []

        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Encoding", unit="batch")
            except ImportError:
                pass  # tqdm not available, proceed without progress bar

        with torch.no_grad():
            for i in iterator:
                batch_sentences = sentences[i : i + batch_size]

                # Tokenize
                inputs = tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Encode
                outputs = self(**inputs)
                embeddings = outputs.pooler_output

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @classmethod
    def from_seq2seq(
        cls,
        seq2seq_model: "Qwen3ForSeq2SeqLM",
        pooling_mode: str = "mean",
        normalize_embeddings: bool = True,
    ) -> "Qwen3StandaloneEncoderModel":
        """
        Extract encoder from a trained Seq2Seq model.

        This is useful for extracting the trained encoder after UL2 training
        to use as a standalone embedding model.

        Args:
            seq2seq_model: Trained Qwen3ForSeq2SeqLM model.
            pooling_mode: Pooling strategy for embeddings.
            normalize_embeddings: Whether to L2-normalize embeddings.

        Returns:
            Standalone encoder model with weights copied from the seq2seq model.

        Example:
            ```python
            from qwen3_encdec import Qwen3ForSeq2SeqLM, Qwen3StandaloneEncoderModel

            # After training
            seq2seq = Qwen3ForSeq2SeqLM.from_pretrained("path/to/trained")
            encoder = Qwen3StandaloneEncoderModel.from_seq2seq(seq2seq)
            encoder.save_pretrained("path/to/encoder")
            ```
        """
        # Create encoder config from seq2seq config
        encoder_config = Qwen3EncoderConfig.from_encoder_decoder_config(
            seq2seq_model.config,
            pooling_mode=pooling_mode,
            normalize_embeddings=normalize_embeddings,
        )

        # Create encoder model
        encoder_model = cls(encoder_config)

        # Copy encoder weights
        encoder_state_dict = seq2seq_model.model.encoder.state_dict()
        encoder_model.encoder.load_state_dict(encoder_state_dict)

        return encoder_model
