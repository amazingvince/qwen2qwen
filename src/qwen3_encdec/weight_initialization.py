"""
Weight initialization utilities for Qwen3 Encoder-Decoder model.

Maps pretrained Qwen3-0.6B weights to the encoder-decoder architecture,
handling the merged attention pattern and embedding extension.
"""

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_encdec import Qwen3ForSeq2SeqLM

logger = logging.getLogger(__name__)


class Qwen3WeightMapper:
    """
    Maps Qwen3 decoder-only weights to encoder-decoder architecture.

    Weight mapping strategy:
    - Encoder attention <- Qwen3 self-attention (same weights, different mask)
    - Decoder attention <- Qwen3 self-attention (merged attention pattern)
    - Encoder MLP <- Qwen3 MLP
    - Decoder MLP <- Qwen3 MLP
    - Shared embeddings <- Qwen3 embeddings (extended with sentinels)
    """

    # Qwen3 layer weight patterns
    ATTENTION_KEYS = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",  # QK-Norm
        "k_norm.weight",  # QK-Norm
    ]

    MLP_KEYS = [
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ]

    NORM_KEYS = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]

    def __init__(
        self,
        qwen3_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        num_sentinel_tokens: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize weight mapper.

        Args:
            qwen3_model_name_or_path: HuggingFace model ID or local path
            num_sentinel_tokens: Number of sentinel tokens to add
            device: Device to load weights on
        """
        self.qwen3_model_name = qwen3_model_name_or_path
        self.num_sentinel_tokens = num_sentinel_tokens
        self.device = device

        self._qwen3_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._qwen3_config = None

    def load_qwen3_weights(self) -> Dict[str, torch.Tensor]:
        """
        Load Qwen3 pretrained weights.

        Returns:
            State dict from Qwen3 model
        """
        if self._qwen3_state_dict is not None:
            return self._qwen3_state_dict

        logger.info(f"Loading Qwen3 weights from {self.qwen3_model_name}")

        # Load config first
        self._qwen3_config = AutoConfig.from_pretrained(self.qwen3_model_name)

        # Load model weights
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            self.qwen3_model_name,
            torch_dtype=torch.float32,  # Load in FP32 for accurate copying
            device_map=self.device,
        )

        self._qwen3_state_dict = qwen3_model.state_dict()

        # Free model memory
        del qwen3_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Loaded {len(self._qwen3_state_dict)} weight tensors")
        return self._qwen3_state_dict

    def get_qwen3_config(self) -> AutoConfig:
        """Get the Qwen3 configuration."""
        if self._qwen3_config is None:
            self._qwen3_config = AutoConfig.from_pretrained(self.qwen3_model_name)
        return self._qwen3_config

    def create_encoder_decoder_config(self) -> Qwen3EncoderDecoderConfig:
        """
        Create encoder-decoder config from Qwen3 config.

        Returns:
            Qwen3EncoderDecoderConfig initialized from Qwen3 settings
        """
        qwen3_config = self.get_qwen3_config()

        return Qwen3EncoderDecoderConfig.from_qwen3_config(
            qwen3_config,
            num_sentinel_tokens=self.num_sentinel_tokens,
        )

    def _map_layer_weights(
        self,
        qwen3_state: Dict[str, torch.Tensor],
        layer_idx: int,
        prefix: str,  # "encoder" or "decoder"
    ) -> Dict[str, torch.Tensor]:
        """
        Map weights for a single transformer layer.

        Args:
            qwen3_state: Source state dict
            layer_idx: Layer index
            prefix: Target prefix ("encoder" or "decoder")

        Returns:
            Mapped weights for this layer
        """
        mapped = {}
        qwen3_layer_prefix = f"model.layers.{layer_idx}"
        # Our model uses model.{encoder|decoder}.layers.X
        target_layer_prefix = f"model.{prefix}.layers.{layer_idx}"

        # Map attention weights - both encoder and decoder use self_attn
        for key in self.ATTENTION_KEYS:
            src_key = f"{qwen3_layer_prefix}.self_attn.{key}"
            if src_key in qwen3_state:
                tgt_key = f"{target_layer_prefix}.self_attn.{key}"
                mapped[tgt_key] = qwen3_state[src_key].clone()

        # Map MLP weights
        for key in self.MLP_KEYS:
            src_key = f"{qwen3_layer_prefix}.mlp.{key}"
            if src_key in qwen3_state:
                tgt_key = f"{target_layer_prefix}.mlp.{key}"
                mapped[tgt_key] = qwen3_state[src_key].clone()

        # Map layer norms
        for key in self.NORM_KEYS:
            src_key = f"{qwen3_layer_prefix}.{key}"
            if src_key in qwen3_state:
                tgt_key = f"{target_layer_prefix}.{key}"
                mapped[tgt_key] = qwen3_state[src_key].clone()

        return mapped

    def _extend_embeddings(
        self,
        embedding_weight: torch.Tensor,
        num_sentinel_tokens: int,
    ) -> torch.Tensor:
        """
        Extend embedding matrix with sentinel token embeddings.

        Sentinel embeddings are initialized from Gaussian with same
        statistics as existing embeddings.

        Args:
            embedding_weight: Original embedding [vocab_size, hidden_size]
            num_sentinel_tokens: Number of sentinels to add

        Returns:
            Extended embedding [vocab_size + num_sentinels, hidden_size]
        """
        if num_sentinel_tokens == 0:
            return embedding_weight.clone()

        original_vocab_size, hidden_size = embedding_weight.shape

        # Compute statistics from existing embeddings
        embed_mean = embedding_weight.mean()
        embed_std = embedding_weight.std()

        logger.info(
            f"Extending embeddings: {original_vocab_size} -> "
            f"{original_vocab_size + num_sentinel_tokens} "
            f"(mean={embed_mean:.4f}, std={embed_std:.4f})"
        )

        # Initialize sentinel embeddings
        sentinel_embeddings = (
            torch.randn(
                num_sentinel_tokens,
                hidden_size,
                dtype=embedding_weight.dtype,
                device=embedding_weight.device,
            )
            * embed_std
            + embed_mean
        )

        # Concatenate
        extended = torch.cat([embedding_weight, sentinel_embeddings], dim=0)

        return extended

    def map_weights(
        self,
        target_model: Qwen3ForSeq2SeqLM,
    ) -> Tuple[int, int]:
        """
        Map Qwen3 weights to encoder-decoder model.

        Args:
            target_model: Target encoder-decoder model to initialize

        Returns:
            Tuple of (num_mapped_weights, num_total_weights)
        """
        qwen3_state = self.load_qwen3_weights()
        qwen3_config = self.get_qwen3_config()

        mapped_state = OrderedDict()
        num_layers = qwen3_config.num_hidden_layers

        # Map encoder layers
        logger.info(f"Mapping {num_layers} encoder layers...")
        for layer_idx in range(num_layers):
            layer_weights = self._map_layer_weights(qwen3_state, layer_idx, "encoder")
            mapped_state.update(layer_weights)

        # Map decoder layers
        logger.info(f"Mapping {num_layers} decoder layers...")
        for layer_idx in range(num_layers):
            layer_weights = self._map_layer_weights(qwen3_state, layer_idx, "decoder")
            mapped_state.update(layer_weights)

        # Map and extend embeddings
        if "model.embed_tokens.weight" in qwen3_state:
            original_embeddings = qwen3_state["model.embed_tokens.weight"]
            extended_embeddings = self._extend_embeddings(
                original_embeddings, self.num_sentinel_tokens
            )
            # Our model uses model.shared for the embedding
            mapped_state["model.shared.weight"] = extended_embeddings

        # Map final layer norm (for both encoder and decoder)
        if "model.norm.weight" in qwen3_state:
            final_norm = qwen3_state["model.norm.weight"].clone()
            mapped_state["model.encoder.norm.weight"] = final_norm
            mapped_state["model.decoder.norm.weight"] = final_norm.clone()

        # Load mapped weights
        logger.info("Loading mapped weights into target model...")
        missing, unexpected = target_model.load_state_dict(mapped_state, strict=False)

        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        # The model should handle embedding tying via _tied_weights_keys
        # But we explicitly ensure they are tied
        target_model.tie_weights()

        return len(mapped_state), len(target_model.state_dict())


def initialize_from_qwen3(
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    num_sentinel_tokens: int = 100,
    device: str = "cpu",
) -> Qwen3ForSeq2SeqLM:
    """
    Create and initialize encoder-decoder model from Qwen3 checkpoint.

    This is the main entry point for weight initialization.

    Args:
        model_name_or_path: Qwen3 model to load
        num_sentinel_tokens: Number of sentinel tokens for UL2
        device: Device to load on

    Returns:
        Initialized Qwen3ForSeq2SeqLM model

    Example:
        >>> model = initialize_from_qwen3("Qwen/Qwen3-0.6B")
        >>> # Model is ready for training
    """
    mapper = Qwen3WeightMapper(
        qwen3_model_name_or_path=model_name_or_path,
        num_sentinel_tokens=num_sentinel_tokens,
        device=device,
    )

    # Create config
    config = mapper.create_encoder_decoder_config()

    # Create model
    logger.info("Creating encoder-decoder model...")
    model = Qwen3ForSeq2SeqLM(config)

    # Map weights
    num_mapped, num_total = mapper.map_weights(model)
    logger.info(f"Mapped {num_mapped} weights to {num_total} total parameters")

    return model


def verify_weight_initialization(
    model: Qwen3ForSeq2SeqLM,
    qwen3_model_name: str = "Qwen/Qwen3-0.6B",
    tolerance: float = 1e-5,
) -> Dict[str, bool]:
    """
    Verify that weights were correctly mapped from Qwen3.

    Args:
        model: Initialized encoder-decoder model
        qwen3_model_name: Original Qwen3 model for comparison
        tolerance: Numerical tolerance for comparison

    Returns:
        Dictionary of verification results
    """
    results = {}

    # Load original weights
    qwen3_model = AutoModelForCausalLM.from_pretrained(
        qwen3_model_name,
        torch_dtype=torch.float32,
    )
    qwen3_state = qwen3_model.state_dict()

    # Check encoder layer 0 attention
    enc_q = model.model.encoder.layers[0].self_attn.q_proj.weight
    qwen3_q = qwen3_state["model.layers.0.self_attn.q_proj.weight"]
    results["encoder_attention_match"] = torch.allclose(enc_q, qwen3_q, atol=tolerance)

    # Check decoder layer 0 attention
    dec_q = model.model.decoder.layers[0].self_attn.q_proj.weight
    results["decoder_attention_match"] = torch.allclose(dec_q, qwen3_q, atol=tolerance)

    # Check MLP weights
    enc_gate = model.model.encoder.layers[0].mlp.gate_proj.weight
    qwen3_gate = qwen3_state["model.layers.0.mlp.gate_proj.weight"]
    results["mlp_match"] = torch.allclose(enc_gate, qwen3_gate, atol=tolerance)

    # Check embedding dimensions
    original_vocab = qwen3_state["model.embed_tokens.weight"].shape[0]
    new_vocab = model.model.shared.weight.shape[0]
    results["embedding_extended"] = (
        new_vocab == original_vocab + model.config.num_sentinel_tokens
    )

    # Check embedding tying
    results["embeddings_tied"] = (
        model.model.shared.weight.data_ptr()
        == model.model.encoder.embed_tokens.weight.data_ptr()
        == model.model.decoder.embed_tokens.weight.data_ptr()
    )

    # Check original embeddings preserved
    original_embeds = qwen3_state["model.embed_tokens.weight"]
    new_embeds = model.model.shared.weight[:original_vocab]
    results["original_embeddings_preserved"] = torch.allclose(
        original_embeds, new_embeds, atol=tolerance
    )

    del qwen3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def verify_gradient_flow(
    model: Qwen3ForSeq2SeqLM,
    batch_size: int = 2,
    enc_seq_len: int = 32,
    dec_seq_len: int = 16,
) -> Dict[str, bool]:
    """
    Verify that gradients flow correctly through the model.

    Critical check: Encoder must receive gradients through decoder
    via the merged attention mechanism.

    Args:
        model: Model to verify
        batch_size: Test batch size
        enc_seq_len: Encoder sequence length
        dec_seq_len: Decoder sequence length

    Returns:
        Dictionary of gradient flow verification results
    """
    model.train()
    results = {}

    # Create dummy inputs
    vocab_size = model.config.vocab_size
    encoder_input_ids = torch.randint(0, vocab_size - 100, (batch_size, enc_seq_len))
    decoder_input_ids = torch.randint(0, vocab_size - 100, (batch_size, dec_seq_len))
    labels = torch.randint(0, vocab_size - 100, (batch_size, dec_seq_len))

    # Move to same device as model
    device = next(model.parameters()).device
    encoder_input_ids = encoder_input_ids.to(device)
    decoder_input_ids = decoder_input_ids.to(device)
    labels = labels.to(device)

    # Forward pass with loss
    outputs = model(
        input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
    )

    # Backward pass
    outputs.loss.backward()

    # Check encoder gradients
    encoder_has_gradients = False
    for name, param in model.model.encoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            encoder_has_gradients = True
            break
    results["encoder_receives_gradients"] = encoder_has_gradients

    # Check decoder gradients
    decoder_has_gradients = False
    for name, param in model.model.decoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            decoder_has_gradients = True
            break
    results["decoder_receives_gradients"] = decoder_has_gradients

    # Check embedding gradients
    results["shared_embedding_gradients"] = (
        model.model.shared.weight.grad is not None
        and model.model.shared.weight.grad.abs().sum() > 0
    )

    # Check that encoder attention receives gradients
    enc_attn_grad = model.model.encoder.layers[0].self_attn.q_proj.weight.grad
    results["encoder_attention_gradients"] = (
        enc_attn_grad is not None and enc_attn_grad.abs().sum() > 0
    )

    # Zero gradients for cleanliness
    model.zero_grad()

    return results
