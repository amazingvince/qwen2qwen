"""Qwen3 Seq2Seq model for encoder-decoder tasks.

This module implements the combined encoder-decoder model following T5-Gemma 2
patterns with merged attention and tied embeddings.

Architecture:
- Qwen3Seq2SeqModel: Base model that wires encoder + decoder with shared embeddings
- Qwen3ForSeq2SeqLM: Adds LM head, loss computation, and generation support
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    Seq2SeqLMOutput,
)
from transformers.utils import logging

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_decoder import Qwen3Decoder
from .modeling_qwen3_encoder import Qwen3Encoder, Qwen3EncoderOutput

logger = logging.get_logger(__name__)


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
class Qwen3Seq2SeqModelOutput(ModelOutput):
    """
    Output type for Qwen3Seq2SeqModel (base model without LM head).

    Args:
        last_hidden_state: Decoder's last layer hidden states [batch, dec_len, hidden_size].
        past_key_values: Cached KV states for incremental decoding.
        decoder_hidden_states: All decoder hidden states if output_hidden_states=True.
        decoder_attentions: All decoder attention weights if output_attentions=True.
        encoder_last_hidden_state: Encoder's last layer hidden states.
        encoder_hidden_states: All encoder hidden states if output_hidden_states=True.
        encoder_attentions: All encoder attention weights if output_attentions=True.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# =============================================================================
# Base Seq2Seq Model
# =============================================================================


class Qwen3Seq2SeqPreTrainedModel(PreTrainedModel):
    """
    Base class for Qwen3 Seq2Seq models.

    Provides weight initialization and config handling.
    """

    config_class = Qwen3EncoderDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3EncoderLayer", "Qwen3DecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following Qwen3 conventions."""
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Seq2SeqModel(Qwen3Seq2SeqPreTrainedModel):
    """
    Base encoder-decoder model with shared embeddings.

    This model wires encoder and decoder together with tied embeddings,
    following the T5-Gemma 2 architecture. It returns decoder hidden states
    without the LM head.

    Can accept pre-built encoder/decoder or create them from config.

    Args:
        config: Model configuration.
        encoder: Optional pre-built encoder. Created from config if None.
        decoder: Optional pre-built decoder. Created from config if None.

    Example:
        ```python
        # Create from config
        model = Qwen3Seq2SeqModel(config)

        # Or with pre-built components
        encoder = Qwen3Encoder(config)
        decoder = Qwen3Decoder(config)
        model = Qwen3Seq2SeqModel(config, encoder=encoder, decoder=decoder)

        # Forward pass
        outputs = model(input_ids=enc_ids, decoder_input_ids=dec_ids)
        hidden_states = outputs.last_hidden_state
        ```
    """

    def __init__(
        self,
        config: Qwen3EncoderDecoderConfig,
        encoder: Optional[Qwen3Encoder] = None,
        decoder: Optional[Qwen3Decoder] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)

        # Shared embeddings (T5-Gemma 2: tied across encoder/decoder)
        self.shared = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )

        # Use provided components or create new ones
        self.encoder = encoder if encoder is not None else Qwen3Encoder(config)
        self.decoder = decoder if decoder is not None else Qwen3Decoder(config)

        # Tie embeddings - both encoder and decoder use the shared embedding
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return shared embeddings."""
        return self.shared

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set shared embeddings and propagate to encoder/decoder."""
        self.shared = value
        self.encoder.embed_tokens = value
        self.decoder.embed_tokens = value

    def get_encoder(self) -> Qwen3Encoder:
        """Return encoder."""
        return self.encoder

    def get_decoder(self) -> Qwen3Decoder:
        """Return decoder."""
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, Qwen3EncoderOutput]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen3Seq2SeqModelOutput]:
        """
        Forward pass for the base seq2seq model.

        Args:
            input_ids: Encoder input token IDs [batch, enc_len].
            attention_mask: Encoder attention mask [batch, enc_len].
            decoder_input_ids: Decoder input token IDs [batch, dec_len].
            decoder_attention_mask: Decoder attention mask [batch, dec_len].
            encoder_outputs: Pre-computed encoder outputs (for generation efficiency).
            past_key_values: Cached KV states for incremental decoding.
            inputs_embeds: Pre-computed encoder embeddings (alternative to input_ids).
            decoder_inputs_embeds: Pre-computed decoder embeddings.
            use_cache: Whether to return KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dataclass.

        Returns:
            Qwen3Seq2SeqModelOutput with decoder hidden states and optional fields.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

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
            # Convert tuple to dataclass
            encoder_outputs = Qwen3EncoderOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state

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

        if not return_dict:
            return (
                decoder_outputs.last_hidden_state,
                decoder_outputs.past_key_values,
                decoder_outputs.hidden_states,
                decoder_outputs.attentions,
                encoder_outputs.last_hidden_state,
                encoder_outputs.hidden_states,
                encoder_outputs.attentions,
            )

        return Qwen3Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# =============================================================================
# Seq2Seq LM Model
# =============================================================================


class Qwen3ForSeq2SeqLM(Qwen3Seq2SeqPreTrainedModel, GenerationMixin):
    """
    Qwen3 Seq2Seq model with LM head for language modeling.

    This model wraps Qwen3Seq2SeqModel and adds:
    - LM head for next-token prediction (tied to shared embeddings)
    - Cross-entropy loss computation
    - HuggingFace generate() support

    Args:
        config: Model configuration.
        encoder: Optional pre-built encoder.
        decoder: Optional pre-built decoder.

    Example:
        ```python
        model = Qwen3ForSeq2SeqLM(config)

        # Training with labels
        outputs = model(input_ids=enc_ids, labels=target_ids)
        loss = outputs.loss

        # Generation
        generated = model.generate(input_ids=enc_ids, max_new_tokens=50)
        ```
    """

    _tied_weights_keys = [
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "lm_head.weight",
    ]

    def __init__(
        self,
        config: Qwen3EncoderDecoderConfig,
        encoder: Optional[Qwen3Encoder] = None,
        decoder: Optional[Qwen3Decoder] = None,
    ) -> None:
        super().__init__(config)
        self.config = config

        # Base seq2seq model
        self.model = Qwen3Seq2SeqModel(config, encoder=encoder, decoder=decoder)

        # LM head (tied to shared embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie LM head to shared embeddings
        self.lm_head.weight = self.model.shared.weight

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return shared embeddings."""
        return self.model.shared

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set shared embeddings."""
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        """Return LM head."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Set LM head."""
        self.lm_head = value

    def get_encoder(self) -> Qwen3Encoder:
        """Return encoder for generation."""
        return self.model.encoder

    def get_decoder(self) -> Qwen3Decoder:
        """Return decoder for generation."""
        return self.model.decoder

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

        # Use pad_token_id as fallback for decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id
        if decoder_start_token_id is None:
            raise ValueError(
                "decoder_start_token_id or pad_token_id must be defined in config"
            )

        # Shift right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # Replace -100 (label ignore index) with pad_token_id
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
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
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
        return_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Prepare decoder inputs from labels if needed
        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            decoder_input_ids = self._shift_right(labels)

        # Get decoder hidden states from base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Compute loss and logits
        loss = None
        lm_logits = None

        if labels is not None:
            # Check if we should use Cut Cross Entropy
            use_cce = getattr(self.config, "use_cut_cross_entropy", False)

            if use_cce:
                try:
                    from cut_cross_entropy import linear_cross_entropy

                    # CCE computes loss directly from hidden states and lm_head weight
                    # This avoids materializing the full [batch*seq, vocab_size] logits tensor
                    # CCE requires bf16 or fp16 for backward pass
                    hidden_states = outputs.last_hidden_state
                    classifier_weight = self.lm_head.weight
                    if hidden_states.dtype == torch.float32:
                        hidden_states = hidden_states.bfloat16()
                        classifier_weight = classifier_weight.bfloat16()

                    loss = linear_cross_entropy(
                        hidden_states.view(-1, self.config.hidden_size),
                        classifier_weight,  # [vocab_size, hidden_size]
                        labels.view(-1),
                        ignore_index=-100,
                    )
                    # Only compute logits if needed for output
                    if not return_dict or output_attentions or output_hidden_states:
                        lm_logits = self.lm_head(outputs.last_hidden_state)
                except ImportError:
                    logger.warning_once(
                        "Cut Cross Entropy not available, falling back to standard CE. "
                        "Install with: pip install 'cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git'"
                    )
                    lm_logits = self.lm_head(outputs.last_hidden_state)
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        lm_logits.view(-1, self.config.vocab_size),
                        labels.view(-1),
                    )
            else:
                # Standard cross entropy
                lm_logits = self.lm_head(outputs.last_hidden_state)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )
        else:
            # No labels - just compute logits for inference
            lm_logits = self.lm_head(outputs.last_hidden_state)

        if not return_dict:
            output = (lm_logits,) + (
                outputs.past_key_values,
                outputs.decoder_hidden_states,
                outputs.decoder_attentions,
                outputs.encoder_last_hidden_state,
                outputs.encoder_hidden_states,
                outputs.encoder_attentions,
            )
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=None,  # Not applicable for merged attention
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
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

        This method is called by generate() at each decoding step.

        Args:
            decoder_input_ids: Current decoder tokens.
            past_key_values: Cached KV states.
            attention_mask: Encoder attention mask.
            encoder_outputs: Encoder outputs.
            **kwargs: Additional arguments.

        Returns:
            Dict of model inputs for this generation step.
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
        """Shift labels right for decoder input during training."""
        return self._shift_right(labels)

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
        beam_idx: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Reorder cache for beam search.

        The cache structure is: (dec_k, dec_v, enc_k, enc_v) per layer.

        Args:
            past_key_values: Cached KV states.
            beam_idx: Beam indices for reordering.

        Returns:
            Reordered cache.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # layer_past is (dec_k, dec_v, enc_k, enc_v)
            dec_k, dec_v, enc_k, enc_v = layer_past
            reordered_layer = (
                dec_k.index_select(0, beam_idx.to(dec_k.device)),
                dec_v.index_select(0, beam_idx.to(dec_v.device)),
                enc_k.index_select(0, beam_idx.to(enc_k.device)),
                enc_v.index_select(0, beam_idx.to(enc_v.device)),
            )
            reordered_past += (reordered_layer,)
        return reordered_past

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resize token embeddings and update tied weights.

        Args:
            new_num_tokens: New vocabulary size.
            pad_to_multiple_of: Pad to multiple of this value.

        Returns:
            New embedding layer.
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # Update config
        self.config.vocab_size = new_num_tokens

        # Re-tie LM head
        self.lm_head.weight = self.model.shared.weight

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
        new_embeddings = nn.Embedding(
            new_num_tokens,
            embedding_dim,
            padding_idx=old_embeddings.padding_idx,
        )
        new_embeddings.to(
            old_embeddings.weight.device, dtype=old_embeddings.weight.dtype
        )

        # Initialize new embeddings
        self._init_weights(new_embeddings)

        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy] = old_embeddings.weight.data[
            :num_tokens_to_copy
        ]

        return new_embeddings
