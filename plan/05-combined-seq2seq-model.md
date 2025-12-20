# Story 05: Qwen3ForSeq2SeqLM Combined Model

## Overview

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-005 |
| **Title** | Qwen3ForSeq2SeqLM Combined Model |
| **Priority** | P0 - Critical Path |
| **Estimated Effort** | 3-4 days |
| **Dependencies** | Story 01-04 (Config, Tokenizer, Encoder, Decoder) |
| **Deliverables** | `Qwen3ForSeq2SeqLM`, tied embeddings, LM head, generation support |

---

## Objective

Implement the complete encoder-decoder model class that:
1. Combines the encoder and decoder into a unified model
2. Implements tied embeddings (encoder input, decoder input, output projection)
3. Provides the language modeling head for next-token prediction
4. Supports HuggingFace's `generate()` API for inference
5. Computes cross-entropy loss for training

---

## Background & Context

### Model Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Qwen3ForSeq2SeqLM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌─────────────────┐                                                   │
│    │ shared_embeddings│──────────────────────────────────────────┐       │
│    │ (vocab_size, D)  │                                          │       │
│    └────────┬─────────┘                                          │       │
│             │                                                    │       │
│    ┌────────▼─────────┐    ┌─────────────────────────────────┐  │       │
│    │                  │    │                                 │  │       │
│    │     Encoder      │───▶│           Decoder               │  │       │
│    │ (bidirectional)  │    │   (merged attention)            │  │       │
│    │                  │    │                                 │  │       │
│    └──────────────────┘    └────────────────┬────────────────┘  │       │
│                                             │                   │       │
│                                    ┌────────▼────────┐          │       │
│                                    │   LM Head       │◀─────────┘       │
│                                    │ (weight tied)   │                  │
│                                    └────────┬────────┘                  │
│                                             │                           │
│                                    ┌────────▼────────┐                  │
│                                    │     logits      │                  │
│                                    │  [B, T, vocab]  │                  │
│                                    └─────────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tied Embeddings
T5Gemma 2 uses tied embeddings for parameter efficiency:
- Single embedding matrix serves encoder input, decoder input, and output projection
- ~10.5% parameter savings
- No quality degradation

---

## Technical Requirements

### 1. Main Model Class

#### File: `modeling_qwen3_encdec.py`

```python
"""Qwen3 Encoder-Decoder model for sequence-to-sequence tasks."""

from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.utils import logging

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_encoder import Qwen3Encoder, Qwen3EncoderOutput
from .modeling_qwen3_decoder import Qwen3Decoder, Qwen3DecoderOutput

logger = logging.get_logger(__name__)


class Qwen3ForSeq2SeqLM(PreTrainedModel, GenerationMixin):
    """
    Qwen3-based encoder-decoder model for sequence-to-sequence tasks.
    
    This model combines a bidirectional encoder with a decoder using
    merged self/cross attention, following the T5Gemma 2 architecture.
    
    Args:
        config: Model configuration.
    
    Example:
        ```python
        from qwen3_encdec import Qwen3ForSeq2SeqLM, Qwen3EncoderDecoderConfig
        
        config = Qwen3EncoderDecoderConfig()
        model = Qwen3ForSeq2SeqLM(config)
        
        # Training
        outputs = model(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        loss = outputs.loss
        
        # Generation
        generated = model.generate(input_ids=encoder_input_ids)
        ```
    """
    
    config_class = Qwen3EncoderDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    
    def __init__(self, config: Qwen3EncoderDecoderConfig):
        super().__init__(config)
        self.config = config
        
        # Shared embeddings
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Encoder and decoder
        self.encoder = Qwen3Encoder(config)
        self.decoder = Qwen3Decoder(config)
        
        # LM head (will be tied to shared embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings
        self._tie_embeddings()
        
        # Initialize weights
        self.post_init()
    
    def _tie_embeddings(self):
        """Tie encoder, decoder, and LM head embeddings."""
        # Set encoder and decoder embeddings to shared
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
        
        # Tie LM head to shared embeddings
        self.lm_head.weight = self.shared.weight
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Return shared embeddings."""
        return self.shared
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set shared embeddings and propagate to encoder/decoder."""
        self.shared = value
        self.encoder.embed_tokens = value
        self.decoder.embed_tokens = value
    
    def get_output_embeddings(self) -> nn.Linear:
        """Return LM head."""
        return self.lm_head
    
    def set_output_embeddings(self, value: nn.Linear):
        """Set LM head."""
        self.lm_head = value
    
    def get_encoder(self) -> Qwen3Encoder:
        """Return encoder for generation."""
        return self.encoder
    
    def get_decoder(self) -> Qwen3Decoder:
        """Return decoder for generation."""
        return self.decoder
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Shift input ids one token to the right for teacher forcing.
        
        Args:
            input_ids: Target token IDs [batch, seq_len].
            
        Returns:
            Shifted IDs with decoder_start_token_id prepended.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        
        if decoder_start_token_id is None:
            raise ValueError(
                "decoder_start_token_id must be defined in config for shift_right"
            )
        
        # Shift right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        
        # Replace any -100 values (label ignore index) with pad_token_id
        if pad_token_id is not None:
            shifted_input_ids = shifted_input_ids.masked_fill(
                shifted_input_ids == -100, pad_token_id
            )
        
        return shifted_input_ids
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        Forward pass for sequence-to-sequence language modeling.
        
        Args:
            input_ids: Encoder input token IDs [batch, enc_len].
            attention_mask: Encoder attention mask [batch, enc_len].
            decoder_input_ids: Decoder input token IDs [batch, dec_len].
            decoder_attention_mask: Decoder attention mask [batch, dec_len].
            encoder_outputs: Pre-computed encoder outputs (for generation).
            past_key_values: Cached KV states for incremental decoding.
            inputs_embeds: Pre-computed encoder embeddings.
            decoder_inputs_embeds: Pre-computed decoder embeddings.
            labels: Target token IDs for loss computation [batch, dec_len].
            use_cache: Whether to use/return KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dataclass.
            
        Returns:
            Seq2SeqLMOutput with loss, logits, and optional fields.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_tensors
        
        # Encode if encoder_outputs not provided
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = Qwen3EncoderOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Prepare decoder inputs
        if labels is not None and decoder_input_ids is None:
            # Shift labels right for teacher forcing
            decoder_input_ids = self._shift_right(labels)
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Project to vocabulary
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten logits and labels for loss computation
            loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=None,  # Not applicable for merged attention
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation step.
        
        This method is called by generate() at each generation step.
        
        Args:
            decoder_input_ids: Current decoder tokens.
            past_key_values: Cached KV states.
            attention_mask: Encoder attention mask.
            encoder_outputs: Encoder outputs.
            **kwargs: Additional arguments.
            
        Returns:
            Dict of model inputs.
        """
        # If past is defined, only use the last token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        return {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
    
    def prepare_decoder_input_ids_from_labels(
        self, labels: torch.Tensor
    ) -> torch.Tensor:
        """Shift labels right for decoder input."""
        return self._shift_right(labels)
    
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]],
        beam_idx: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Reorder cache for beam search.
        
        Args:
            past_key_values: Cached KV states.
            beam_idx: Beam indices for reordering.
            
        Returns:
            Reordered cache.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past
                ),
            )
        return reordered_past
    
    def resize_token_embeddings(
        self, new_num_tokens: int
    ) -> nn.Embedding:
        """
        Resize token embeddings and update tied weights.
        
        Args:
            new_num_tokens: New vocabulary size.
            
        Returns:
            New embedding layer.
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        
        # Update config
        self.config.vocab_size = new_num_tokens
        
        # Re-tie weights
        self._tie_embeddings()
        
        return new_embeddings
    
    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: int,
    ) -> nn.Embedding:
        """Create resized embedding layer."""
        old_num_tokens, embedding_dim = old_embeddings.weight.shape
        
        if new_num_tokens == old_num_tokens:
            return old_embeddings
        
        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Initialize new embeddings
        self._init_weights(new_embeddings)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy] = old_embeddings.weight.data[:num_tokens_to_copy]
        
        return new_embeddings


# Encoder-only wrapper for embedding extraction
class Qwen3EncoderModel(PreTrainedModel):
    """
    Encoder-only model for embedding extraction.
    
    This is useful for extracting trained encoder weights after UL2 training.
    
    Args:
        config: Model configuration.
    """
    
    config_class = Qwen3EncoderDecoderConfig
    
    def __init__(self, config: Qwen3EncoderDecoderConfig):
        super().__init__(config)
        self.encoder = Qwen3Encoder(config)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen3EncoderOutput]:
        """Forward pass through encoder only."""
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    @classmethod
    def from_seq2seq(
        cls,
        seq2seq_model: Qwen3ForSeq2SeqLM,
    ) -> "Qwen3EncoderModel":
        """
        Extract encoder from a trained Seq2Seq model.
        
        Args:
            seq2seq_model: Trained Qwen3ForSeq2SeqLM model.
            
        Returns:
            Encoder-only model.
        """
        encoder_model = cls(seq2seq_model.config)
        encoder_model.encoder.load_state_dict(seq2seq_model.encoder.state_dict())
        return encoder_model
```

### 2. Model Registration

Add to `__init__.py`:

```python
"""Qwen3 Encoder-Decoder model implementation."""

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .tokenization_qwen3_encdec import Qwen3EncoderDecoderTokenizer
from .modeling_qwen3_encoder import Qwen3Encoder, Qwen3EncoderOutput
from .modeling_qwen3_decoder import Qwen3Decoder, Qwen3DecoderOutput
from .modeling_qwen3_encdec import (
    Qwen3ForSeq2SeqLM,
    Qwen3EncoderModel,
)

__version__ = "0.1.0"

__all__ = [
    "Qwen3EncoderDecoderConfig",
    "Qwen3EncoderDecoderTokenizer",
    "Qwen3Encoder",
    "Qwen3EncoderOutput",
    "Qwen3Decoder",
    "Qwen3DecoderOutput",
    "Qwen3ForSeq2SeqLM",
    "Qwen3EncoderModel",
]
```

---

## Unit Tests

#### File: `tests/test_model.py`

```python
"""Unit tests for Qwen3ForSeq2SeqLM."""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from qwen3_encdec.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from qwen3_encdec.modeling_qwen3_encdec import (
    Qwen3ForSeq2SeqLM,
    Qwen3EncoderModel,
)


class TestQwen3ForSeq2SeqLM:
    """Test combined Seq2Seq model."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=4,
            intermediate_size=512,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    
    @pytest.fixture
    def model(self, config):
        return Qwen3ForSeq2SeqLM(config)
    
    def test_initialization(self, model, config):
        """Test model initialization."""
        assert model.config == config
        assert model.shared is not None
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.lm_head is not None
    
    def test_tied_embeddings(self, model):
        """Test that embeddings are properly tied."""
        # All should reference same tensor
        assert model.shared.weight is model.encoder.embed_tokens.weight
        assert model.shared.weight is model.decoder.embed_tokens.weight
        assert model.shared.weight is model.lm_head.weight
    
    def test_forward_basic(self, model):
        """Test basic forward pass."""
        input_ids = torch.randint(0, 1000, (2, 10))
        decoder_input_ids = torch.randint(0, 1000, (2, 8))
        
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        )
        
        assert outputs.logits.shape == (2, 8, 1000)
        assert outputs.loss is None  # No labels provided
    
    def test_forward_with_labels(self, model):
        """Test forward pass with labels for training."""
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 8))
        
        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )
        
        assert outputs.loss is not None
        assert outputs.loss.ndim == 0  # Scalar loss
        assert outputs.logits.shape == (2, 8, 1000)
    
    def test_shift_right(self, model, config):
        """Test label shifting for teacher forcing."""
        labels = torch.tensor([[10, 20, 30, 40]])
        
        shifted = model._shift_right(labels)
        
        assert shifted[0, 0] == config.decoder_start_token_id
        assert shifted[0, 1] == 10
        assert shifted[0, 2] == 20
        assert shifted[0, 3] == 30
    
    def test_encoder_outputs_reuse(self, model):
        """Test that encoder outputs can be cached."""
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Get encoder outputs
        encoder_outputs = model.encoder(input_ids)
        
        # Use cached encoder outputs
        decoder_input_ids = torch.randint(0, 1000, (2, 8))
        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )
        
        assert outputs.logits.shape == (2, 8, 1000)
    
    def test_gradient_flow(self, model):
        """Test gradient flow from loss to all parameters."""
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 8))
        
        outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()
        
        # Check encoder gradients
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder.{name}"
        
        # Check decoder gradients
        for name, param in model.decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for decoder.{name}"
        
        # Check shared embeddings gradient
        assert model.shared.weight.grad is not None
    
    def test_output_hidden_states(self, model):
        """Test returning all hidden states."""
        input_ids = torch.randint(0, 1000, (2, 10))
        decoder_input_ids = torch.randint(0, 1000, (2, 8))
        
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
        )
        
        assert outputs.encoder_hidden_states is not None
        assert outputs.decoder_hidden_states is not None
    
    def test_output_attentions(self, model):
        """Test returning attention weights."""
        input_ids = torch.randint(0, 1000, (2, 10))
        decoder_input_ids = torch.randint(0, 1000, (2, 8))
        
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
        )
        
        assert outputs.encoder_attentions is not None
        assert outputs.decoder_attentions is not None
    
    def test_resize_embeddings(self, model):
        """Test vocabulary resizing."""
        new_size = 1100
        model.resize_token_embeddings(new_size)
        
        assert model.shared.weight.shape[0] == new_size
        assert model.lm_head.weight.shape[0] == new_size
        assert model.config.vocab_size == new_size
        
        # Still tied
        assert model.shared.weight is model.lm_head.weight


class TestGeneration:
    """Test generation capabilities."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=0,
        )
    
    @pytest.fixture
    def model(self, config):
        model = Qwen3ForSeq2SeqLM(config)
        model.eval()
        return model
    
    def test_greedy_generation(self, model):
        """Test greedy decoding."""
        input_ids = torch.randint(2, 100, (1, 5))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=False,
            )
        
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 11  # max_new_tokens + decoder_start
    
    def test_sampling_generation(self, model):
        """Test sampling-based generation."""
        input_ids = torch.randint(2, 100, (1, 5))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=True,
                temperature=1.0,
                top_k=50,
            )
        
        assert generated.shape[0] == 1
    
    def test_beam_search_generation(self, model):
        """Test beam search generation."""
        input_ids = torch.randint(2, 100, (1, 5))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                num_beams=3,
                do_sample=False,
            )
        
        assert generated.shape[0] == 1
    
    def test_prepare_inputs_for_generation(self, model):
        """Test generation input preparation."""
        decoder_input_ids = torch.tensor([[0, 10, 20]])
        encoder_outputs = model.encoder(torch.randint(2, 100, (1, 5)))
        
        inputs = model.prepare_inputs_for_generation(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
        )
        
        assert "decoder_input_ids" in inputs
        assert "encoder_outputs" in inputs
        assert inputs["use_cache"] is True
    
    def test_incremental_generation(self, model):
        """Test that KV cache is used correctly."""
        input_ids = torch.randint(2, 100, (1, 5))
        
        with torch.no_grad():
            # Full sequence generation tracks cache usage
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=5,
                use_cache=True,
                do_sample=False,
            )
        
        assert generated.shape[1] <= 6


class TestQwen3EncoderModel:
    """Test encoder-only model."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=4,
            intermediate_size=512,
        )
    
    def test_forward(self, config):
        """Test encoder-only forward pass."""
        model = Qwen3EncoderModel(config)
        input_ids = torch.randint(0, 1000, (2, 10))
        
        outputs = model(input_ids)
        
        assert outputs.last_hidden_state.shape == (2, 10, 256)
    
    def test_from_seq2seq(self, config):
        """Test extracting encoder from Seq2Seq model."""
        # Create and "train" a Seq2Seq model
        seq2seq = Qwen3ForSeq2SeqLM(config)
        
        # Modify a weight to verify transfer
        with torch.no_grad():
            seq2seq.encoder.layers[0].mlp.gate_proj.weight.fill_(0.5)
        
        # Extract encoder
        encoder_model = Qwen3EncoderModel.from_seq2seq(seq2seq)
        
        # Verify weights transferred
        assert torch.allclose(
            encoder_model.encoder.layers[0].mlp.gate_proj.weight,
            seq2seq.encoder.layers[0].mlp.gate_proj.weight,
        )


class TestModelSaveLoad:
    """Test model serialization."""
    
    @pytest.fixture
    def config(self):
        return Qwen3EncoderDecoderConfig(
            vocab_size=1000,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            decoder_start_token_id=0,
            pad_token_id=0,
        )
    
    def test_save_and_load(self, config):
        """Test model save and load."""
        model = Qwen3ForSeq2SeqLM(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_pretrained(tmpdir)
            
            # Check files exist
            assert (Path(tmpdir) / "config.json").exists()
            assert (Path(tmpdir) / "model.safetensors").exists() or \
                   (Path(tmpdir) / "pytorch_model.bin").exists()
            
            # Load
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)
            
            # Verify config
            assert loaded.config.hidden_size == config.hidden_size
            
            # Verify weights (spot check)
            assert torch.allclose(
                model.shared.weight,
                loaded.shared.weight,
            )
    
    def test_tied_weights_after_load(self, config):
        """Test that weights remain tied after load."""
        model = Qwen3ForSeq2SeqLM(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)
            
            # Check weights are tied
            assert loaded.shared.weight is loaded.encoder.embed_tokens.weight
            assert loaded.shared.weight is loaded.decoder.embed_tokens.weight
            assert loaded.shared.weight is loaded.lm_head.weight
```

---

## Acceptance Criteria

1. **Forward Pass**: Model computes logits from encoder input + decoder input
2. **Loss Computation**: Cross-entropy loss computed correctly with labels
3. **Label Shifting**: `_shift_right()` correctly prepares decoder inputs
4. **Tied Embeddings**: Single embedding matrix shared across encoder, decoder, LM head
5. **Generation**: `model.generate()` works with greedy, sampling, and beam search
6. **KV Cache**: Incremental decoding works correctly
7. **Encoder Caching**: Can reuse encoder outputs for multiple generations
8. **Save/Load**: Model serialization preserves weights and tied embeddings
9. **Encoder Extraction**: `Qwen3EncoderModel.from_seq2seq()` works
10. **Gradient Flow**: Gradients propagate through entire model
11. **Unit Tests**: All tests pass with >95% coverage

---

## Notes for Developer

1. **Generation Config**: You may need to create a `GenerationConfig` for better control over generation parameters.

2. **Beam Search Cache Reordering**: The `_reorder_cache` method is critical for beam search - test it thoroughly.

3. **Mixed Precision**: Ensure the model works with BF16/FP16 during training and inference.

4. **Flash Attention**: Consider adding Flash Attention support for efficiency.

5. **Padding Strategy**: Verify the padding behavior matches expectations for batched generation.

---

## Next Story

After completing this story, proceed to **Story 06: Weight Initialization from Qwen3-0.6B**.
