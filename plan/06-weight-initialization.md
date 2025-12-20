# Story 06: Weight Initialization from Qwen3 Checkpoint

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-06 |
| **Title** | Weight Initialization from Qwen3-0.6B Checkpoint |
| **Epic** | Qwen3 Encoder-Decoder Implementation |
| **Priority** | High |
| **Estimated Effort** | 2-3 days |
| **Dependencies** | Stories 01-05 (Config, Tokenizer, Encoder, Decoder, Seq2SeqLM) |

---

## Objective

Implement a weight initialization system that loads pretrained weights from Qwen3-0.6B and correctly maps them to both the encoder and decoder of the new encoder-decoder model. This includes handling the merged attention weight mapping and extending embeddings with sentinel tokens.

---

## Background

The T5Gemma 2 approach reuses decoder-only pretrained weights for both encoder and decoder:

1. **Encoder**: Uses the same attention weights but with bidirectional attention mask
2. **Decoder**: Uses the same attention weights in merged attention configuration
3. **Embeddings**: Extended with sentinel tokens, initialized from existing embedding statistics

This preserves the pretrained knowledge from Qwen3-0.6B while adapting the architecture.

---

## Technical Requirements

### 6.1 Weight Mapping Utility

```python
# src/weight_initialization.py

"""
Weight initialization utilities for Qwen3 Encoder-Decoder model.

Maps pretrained Qwen3-0.6B weights to the encoder-decoder architecture,
handling the merged attention pattern and embedding extension.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging
from collections import OrderedDict

from transformers import AutoModelForCausalLM, AutoConfig
from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .modeling_qwen3_encdec import Qwen3ForSeq2SeqLM

logger = logging.getLogger(__name__)


class Qwen3WeightMapper:
    """
    Maps Qwen3 decoder-only weights to encoder-decoder architecture.
    
    Weight mapping strategy:
    - Encoder attention <- Qwen3 self-attention (same weights, different mask)
    - Decoder merged attention <- Qwen3 self-attention (same Q, K, V, O projections)
    - Encoder MLP <- Qwen3 MLP
    - Decoder MLP <- Qwen3 MLP
    - Shared embeddings <- Qwen3 embeddings (extended with sentinels)
    """
    
    # Qwen3 layer weight patterns
    QWEN3_ATTENTION_KEYS = [
        "q_proj.weight",
        "k_proj.weight", 
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",  # QK-Norm
        "k_norm.weight",  # QK-Norm
    ]
    
    QWEN3_MLP_KEYS = [
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ]
    
    QWEN3_NORM_KEYS = [
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
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
        target_layer_prefix = f"{prefix}.layers.{layer_idx}"
        
        # Map attention weights
        attention_target = "self_attn" if prefix == "encoder" else "merged_attn"
        for key in self.QWEN3_ATTENTION_KEYS:
            src_key = f"{qwen3_layer_prefix}.self_attn.{key}"
            if src_key in qwen3_state:
                tgt_key = f"{target_layer_prefix}.{attention_target}.{key}"
                mapped[tgt_key] = qwen3_state[src_key].clone()
        
        # Map MLP weights
        for key in self.QWEN3_MLP_KEYS:
            src_key = f"{qwen3_layer_prefix}.mlp.{key}"
            if src_key in qwen3_state:
                tgt_key = f"{target_layer_prefix}.mlp.{key}"
                mapped[tgt_key] = qwen3_state[src_key].clone()
        
        # Map layer norms
        for key in self.QWEN3_NORM_KEYS:
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
        sentinel_embeddings = torch.randn(
            num_sentinel_tokens, 
            hidden_size,
            dtype=embedding_weight.dtype,
            device=embedding_weight.device,
        ) * embed_std + embed_mean
        
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
            layer_weights = self._map_layer_weights(
                qwen3_state, layer_idx, "encoder"
            )
            mapped_state.update(layer_weights)
        
        # Map decoder layers
        logger.info(f"Mapping {num_layers} decoder layers...")
        for layer_idx in range(num_layers):
            layer_weights = self._map_layer_weights(
                qwen3_state, layer_idx, "decoder"
            )
            mapped_state.update(layer_weights)
        
        # Map and extend embeddings
        if "model.embed_tokens.weight" in qwen3_state:
            original_embeddings = qwen3_state["model.embed_tokens.weight"]
            extended_embeddings = self._extend_embeddings(
                original_embeddings, self.num_sentinel_tokens
            )
            mapped_state["shared.weight"] = extended_embeddings
        
        # Map final layer norm (for both encoder and decoder)
        if "model.norm.weight" in qwen3_state:
            final_norm = qwen3_state["model.norm.weight"].clone()
            mapped_state["encoder.final_layernorm.weight"] = final_norm
            mapped_state["decoder.final_layernorm.weight"] = final_norm.clone()
        
        # Load mapped weights
        logger.info("Loading mapped weights into target model...")
        missing, unexpected = target_model.load_state_dict(
            mapped_state, strict=False
        )
        
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        
        # Tie embeddings
        target_model._tie_embeddings()
        
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
    enc_q = model.encoder.layers[0].self_attn.q_proj.weight
    qwen3_q = qwen3_state["model.layers.0.self_attn.q_proj.weight"]
    results["encoder_attention_match"] = torch.allclose(
        enc_q, qwen3_q, atol=tolerance
    )
    
    # Check decoder layer 0 attention
    dec_q = model.decoder.layers[0].merged_attn.q_proj.weight
    results["decoder_attention_match"] = torch.allclose(
        dec_q, qwen3_q, atol=tolerance
    )
    
    # Check MLP weights
    enc_gate = model.encoder.layers[0].mlp.gate_proj.weight
    qwen3_gate = qwen3_state["model.layers.0.mlp.gate_proj.weight"]
    results["mlp_match"] = torch.allclose(
        enc_gate, qwen3_gate, atol=tolerance
    )
    
    # Check embedding dimensions
    original_vocab = qwen3_state["model.embed_tokens.weight"].shape[0]
    new_vocab = model.shared.weight.shape[0]
    results["embedding_extended"] = (
        new_vocab == original_vocab + model.config.num_sentinel_tokens
    )
    
    # Check embedding tying
    results["embeddings_tied"] = (
        model.shared.weight.data_ptr() == 
        model.encoder.embed_tokens.weight.data_ptr() ==
        model.decoder.embed_tokens.weight.data_ptr()
    )
    
    # Check original embeddings preserved
    original_embeds = qwen3_state["model.embed_tokens.weight"]
    new_embeds = model.shared.weight[:original_vocab]
    results["original_embeddings_preserved"] = torch.allclose(
        original_embeds, new_embeds, atol=tolerance
    )
    
    del qwen3_model
    
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
    encoder_input_ids = torch.randint(0, vocab_size, (batch_size, enc_seq_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, dec_seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, dec_seq_len))
    
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
    for name, param in model.encoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            encoder_has_gradients = True
            break
    results["encoder_receives_gradients"] = encoder_has_gradients
    
    # Check decoder gradients
    decoder_has_gradients = False
    for name, param in model.decoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            decoder_has_gradients = True
            break
    results["decoder_receives_gradients"] = decoder_has_gradients
    
    # Check embedding gradients
    results["shared_embedding_gradients"] = (
        model.shared.weight.grad is not None and
        model.shared.weight.grad.abs().sum() > 0
    )
    
    # Check that encoder attention receives gradients
    enc_attn_grad = model.encoder.layers[0].self_attn.q_proj.weight.grad
    results["encoder_attention_gradients"] = (
        enc_attn_grad is not None and enc_attn_grad.abs().sum() > 0
    )
    
    # Zero gradients for cleanliness
    model.zero_grad()
    
    return results
```

### 6.2 Command-Line Initialization Script

```python
# scripts/initialize_model.py

"""
Command-line script to initialize Qwen3 encoder-decoder from checkpoint.

Usage:
    python scripts/initialize_model.py \
        --qwen3-model Qwen/Qwen3-0.6B \
        --output-dir ./qwen3-encdec-initialized \
        --verify
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.weight_initialization import (
    initialize_from_qwen3,
    verify_weight_initialization,
    verify_gradient_flow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Qwen3 encoder-decoder from pretrained checkpoint"
    )
    parser.add_argument(
        "--qwen3-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Qwen3 model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for initialized model",
    )
    parser.add_argument(
        "--num-sentinel-tokens",
        type=int,
        default=100,
        help="Number of sentinel tokens for UL2",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification checks after initialization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for saving",
    )
    
    args = parser.parse_args()
    
    # Initialize model
    logger.info(f"Initializing from {args.qwen3_model}...")
    model = initialize_from_qwen3(
        model_name_or_path=args.qwen3_model,
        num_sentinel_tokens=args.num_sentinel_tokens,
        device=args.device,
    )
    
    # Run verification if requested
    if args.verify:
        logger.info("Running weight verification...")
        weight_results = verify_weight_initialization(model, args.qwen3_model)
        for check, passed in weight_results.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}")
        
        if not all(weight_results.values()):
            logger.error("Weight verification failed!")
            return 1
        
        logger.info("Running gradient flow verification...")
        grad_results = verify_gradient_flow(model)
        for check, passed in grad_results.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}")
        
        if not all(grad_results.values()):
            logger.error("Gradient flow verification failed!")
            return 1
        
        logger.info("All verification checks passed!")
    
    # Convert dtype if needed
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype != "float32":
        model = model.to(dtype_map[args.dtype])
    
    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
    
    # Save initialization info
    init_info = {
        "source_model": args.qwen3_model,
        "num_sentinel_tokens": args.num_sentinel_tokens,
        "dtype": args.dtype,
        "verification_passed": args.verify,
    }
    with open(output_path / "initialization_info.json", "w") as f:
        json.dump(init_info, f, indent=2)
    
    logger.info("Initialization complete!")
    return 0


if __name__ == "__main__":
    exit(main())
```

---

## Unit Tests

```python
# tests/test_weight_initialization.py

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.weight_initialization import (
    Qwen3WeightMapper,
    initialize_from_qwen3,
    verify_weight_initialization,
    verify_gradient_flow,
)
from src.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from src.modeling_qwen3_encdec import Qwen3ForSeq2SeqLM


class TestQwen3WeightMapper:
    """Tests for Qwen3WeightMapper class."""
    
    @pytest.fixture
    def mock_qwen3_state_dict(self):
        """Create mock Qwen3 state dict."""
        hidden_size = 256
        intermediate_size = 512
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden_size // num_heads
        
        state = {}
        
        # Embedding
        state["model.embed_tokens.weight"] = torch.randn(1000, hidden_size)
        
        # One layer for testing
        layer_prefix = "model.layers.0"
        
        # Attention
        state[f"{layer_prefix}.self_attn.q_proj.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        state[f"{layer_prefix}.self_attn.k_proj.weight"] = torch.randn(
            num_kv_heads * head_dim, hidden_size
        )
        state[f"{layer_prefix}.self_attn.v_proj.weight"] = torch.randn(
            num_kv_heads * head_dim, hidden_size
        )
        state[f"{layer_prefix}.self_attn.o_proj.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        state[f"{layer_prefix}.self_attn.q_norm.weight"] = torch.randn(head_dim)
        state[f"{layer_prefix}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        
        # MLP
        state[f"{layer_prefix}.mlp.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state[f"{layer_prefix}.mlp.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state[f"{layer_prefix}.mlp.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size
        )
        
        # Norms
        state[f"{layer_prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.randn(
            hidden_size
        )
        
        # Final norm
        state["model.norm.weight"] = torch.randn(hidden_size)
        
        return state
    
    def test_extend_embeddings_shape(self, mock_qwen3_state_dict):
        """Test embedding extension produces correct shape."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=100)
        
        original = mock_qwen3_state_dict["model.embed_tokens.weight"]
        original_size = original.shape[0]
        
        extended = mapper._extend_embeddings(original, 100)
        
        assert extended.shape == (original_size + 100, original.shape[1])
    
    def test_extend_embeddings_preserves_original(self, mock_qwen3_state_dict):
        """Test that original embeddings are preserved after extension."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=100)
        
        original = mock_qwen3_state_dict["model.embed_tokens.weight"]
        extended = mapper._extend_embeddings(original, 100)
        
        # Original embeddings should be unchanged
        assert torch.allclose(extended[:original.shape[0]], original)
    
    def test_extend_embeddings_sentinel_statistics(self, mock_qwen3_state_dict):
        """Test sentinel embeddings have similar statistics to original."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=1000)  # More samples
        
        original = mock_qwen3_state_dict["model.embed_tokens.weight"]
        extended = mapper._extend_embeddings(original, 1000)
        
        sentinel_part = extended[original.shape[0]:]
        
        # Statistics should be roughly similar (within reasonable tolerance)
        original_std = original.std()
        sentinel_std = sentinel_part.std()
        
        # Allow 20% deviation
        assert abs(sentinel_std - original_std) < 0.2 * original_std
    
    def test_map_layer_weights_encoder(self, mock_qwen3_state_dict):
        """Test layer weight mapping for encoder."""
        mapper = Qwen3WeightMapper()
        
        mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")
        
        # Check encoder attention keys exist
        assert "encoder.layers.0.self_attn.q_proj.weight" in mapped
        assert "encoder.layers.0.self_attn.k_proj.weight" in mapped
        assert "encoder.layers.0.mlp.gate_proj.weight" in mapped
        assert "encoder.layers.0.input_layernorm.weight" in mapped
    
    def test_map_layer_weights_decoder(self, mock_qwen3_state_dict):
        """Test layer weight mapping for decoder with merged attention."""
        mapper = Qwen3WeightMapper()
        
        mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "decoder")
        
        # Check decoder uses merged_attn prefix
        assert "decoder.layers.0.merged_attn.q_proj.weight" in mapped
        assert "decoder.layers.0.merged_attn.k_proj.weight" in mapped
        assert "decoder.layers.0.mlp.gate_proj.weight" in mapped
    
    def test_map_layer_weights_values_match(self, mock_qwen3_state_dict):
        """Test that mapped values match source values."""
        mapper = Qwen3WeightMapper()
        
        enc_mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "encoder")
        dec_mapped = mapper._map_layer_weights(mock_qwen3_state_dict, 0, "decoder")
        
        # Encoder and decoder should have same values (from same source)
        src_q = mock_qwen3_state_dict["model.layers.0.self_attn.q_proj.weight"]
        
        assert torch.allclose(
            enc_mapped["encoder.layers.0.self_attn.q_proj.weight"],
            src_q
        )
        assert torch.allclose(
            dec_mapped["decoder.layers.0.merged_attn.q_proj.weight"],
            src_q
        )


class TestWeightInitializationIntegration:
    """Integration tests for full weight initialization."""
    
    @pytest.fixture
    def small_config(self):
        """Create small config for testing."""
        return Qwen3EncoderDecoderConfig(
            vocab_size=1100,  # 1000 + 100 sentinels
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            num_sentinel_tokens=100,
        )
    
    @pytest.fixture
    def small_model(self, small_config):
        """Create small model for testing."""
        return Qwen3ForSeq2SeqLM(small_config)
    
    def test_verify_gradient_flow_all_pass(self, small_model):
        """Test gradient flow verification with valid model."""
        results = verify_gradient_flow(
            small_model,
            batch_size=2,
            enc_seq_len=16,
            dec_seq_len=8,
        )
        
        # All checks should pass for a properly constructed model
        assert results["encoder_receives_gradients"]
        assert results["decoder_receives_gradients"]
        assert results["shared_embedding_gradients"]
        assert results["encoder_attention_gradients"]
    
    def test_tied_embeddings_after_initialization(self, small_model):
        """Test that embeddings are tied after model creation."""
        # Embeddings should be tied
        assert (
            small_model.shared.weight.data_ptr() ==
            small_model.encoder.embed_tokens.weight.data_ptr()
        )
        assert (
            small_model.shared.weight.data_ptr() ==
            small_model.decoder.embed_tokens.weight.data_ptr()
        )


class TestSaveLoad:
    """Tests for saving and loading initialized models."""
    
    @pytest.fixture
    def small_model(self):
        """Create small model for save/load testing."""
        config = Qwen3EncoderDecoderConfig(
            vocab_size=1100,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            num_sentinel_tokens=100,
        )
        return Qwen3ForSeq2SeqLM(config)
    
    def test_save_and_load_preserves_weights(self, small_model):
        """Test that save/load preserves all weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            small_model.save_pretrained(tmpdir)
            
            # Load
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)
            
            # Compare weights
            for name, param in small_model.named_parameters():
                loaded_param = dict(loaded.named_parameters())[name]
                assert torch.allclose(param, loaded_param), f"Mismatch in {name}"
    
    def test_save_and_load_preserves_tied_embeddings(self, small_model):
        """Test that tied embeddings remain tied after load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_model.save_pretrained(tmpdir)
            loaded = Qwen3ForSeq2SeqLM.from_pretrained(tmpdir)
            
            # Check tying is preserved
            assert (
                loaded.shared.weight.data_ptr() ==
                loaded.encoder.embed_tokens.weight.data_ptr()
            )


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_sentinel_tokens(self):
        """Test with no sentinel tokens."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=0)
        
        embeddings = torch.randn(1000, 256)
        extended = mapper._extend_embeddings(embeddings, 0)
        
        # Should be unchanged
        assert extended.shape == embeddings.shape
        assert torch.allclose(extended, embeddings)
    
    def test_large_sentinel_count(self):
        """Test with many sentinel tokens."""
        mapper = Qwen3WeightMapper(num_sentinel_tokens=1000)
        
        embeddings = torch.randn(100, 256)
        extended = mapper._extend_embeddings(embeddings, 1000)
        
        assert extended.shape == (1100, 256)
```

---

## Acceptance Criteria

1. **Weight Mapping**
   - [ ] All encoder layers receive weights from Qwen3 self-attention
   - [ ] All decoder layers receive weights from Qwen3 self-attention (for merged attention)
   - [ ] MLP weights correctly copied to both encoder and decoder
   - [ ] Layer norms correctly mapped

2. **Embedding Handling**
   - [ ] Original vocabulary embeddings preserved exactly
   - [ ] Sentinel embeddings initialized with matching statistics
   - [ ] Embedding matrix correctly extended to vocab_size + num_sentinels
   - [ ] Embeddings properly tied across encoder, decoder, and LM head

3. **Verification**
   - [ ] Weight verification confirms exact match for non-extended weights
   - [ ] Gradient flow verification confirms encoder receives gradients
   - [ ] All verification checks pass after initialization

4. **Persistence**
   - [ ] Initialized model can be saved with `save_pretrained()`
   - [ ] Loaded model matches saved model exactly
   - [ ] Tied embeddings remain tied after save/load

5. **CLI Tool**
   - [ ] Script successfully initializes from Qwen3-0.6B
   - [ ] Verification flags work correctly
   - [ ] Model saves to specified output directory

---

## Dependencies

- **Story 01**: Configuration class for `Qwen3EncoderDecoderConfig`
- **Story 03**: Encoder implementation for weight loading
- **Story 04**: Decoder implementation for merged attention weights
- **Story 05**: Combined model for full initialization

---

## Estimated Effort

- Implementation: 1.5 days
- Testing: 0.5 days
- Integration testing with real Qwen3 weights: 0.5 days
- **Total: 2-3 days**

---

## Developer Notes

1. **Memory Management**: Loading Qwen3 weights requires significant memory. The implementation deletes the source model after extracting weights to free memory.

2. **Weight Naming**: Pay attention to the exact weight naming conventions in Qwen3's state dict. The mapping must match exactly.

3. **Numerical Precision**: Use FP32 for weight copying to avoid precision loss, then convert to desired dtype afterward.

4. **Verification Importance**: The gradient flow verification is critical - if the encoder doesn't receive gradients through the decoder's merged attention, training won't work properly.

5. **Reference Files**:
   - `transformers/models/qwen3/modeling_qwen3.py` - Source weight structure
   - T5Gemma 2 implementation - Pattern for weight mapping

6. **Common Issues**:
   - Missing QK-Norm weights in older models
   - Different naming for attention vs merged_attn
   - Forgetting to call `_tie_embeddings()` after loading
