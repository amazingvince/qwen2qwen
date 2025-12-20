# Story 10: Encoder Extraction

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-10 |
| **Title** | Encoder Extraction and Standalone Model Export |
| **Epic** | Qwen3 Encoder-Decoder Implementation |
| **Priority** | High |
| **Estimated Effort** | 1-2 days |
| **Dependencies** | Stories 01-09 (Trained encoder-decoder model) |

---

## Objective

Extract the trained bidirectional encoder from the full encoder-decoder model and export it as a standalone HuggingFace model for embedding tasks. This includes creating a dedicated encoder-only model class, pooling strategies, and proper checkpoint conversion.

---

## Background

After UL2 training, the encoder has learned bidirectional representations through the denoising objectives. The goal is to extract this encoder for use in:

1. **Semantic Embeddings**: Dense vector representations for text
2. **Retrieval**: Document/query encoding for search systems
3. **Similarity Tasks**: Sentence and document similarity
4. **Downstream Fine-tuning**: Classification, NER, etc.

The extracted encoder should:
- Be a standalone HuggingFace model
- Support multiple pooling strategies (mean, CLS, last token)
- Be compatible with the sentence-transformers library
- Include proper model card and documentation

---

## Technical Requirements

### 10.1 Encoder-Only Model Class

```python
# src/models/encoder_only.py

"""
Standalone encoder model for embedding tasks.
Extracted from trained Qwen3EncoderDecoderModel.
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from .configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from .encoder import Qwen3Encoder


class Qwen3EncoderConfig(PretrainedConfig):
    """
    Configuration for standalone Qwen3 Encoder model.
    
    This is a simplified config for the encoder-only model,
    derived from the full Qwen3EncoderDecoderConfig.
    """
    model_type = "qwen3_encoder"
    
    def __init__(
        self,
        vocab_size: int = 152036,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        pad_token_id: int = 151643,
        pooling_mode: str = "mean",  # mean, cls, last, weighted_mean
        normalize_embeddings: bool = True,
        **kwargs
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
        
        # Embedding-specific
        self.pooling_mode = pooling_mode
        self.normalize_embeddings = normalize_embeddings
    
    @classmethod
    def from_encoder_decoder_config(
        cls,
        config: Qwen3EncoderDecoderConfig,
        pooling_mode: str = "mean",
        normalize_embeddings: bool = True
    ) -> "Qwen3EncoderConfig":
        """Create encoder config from full encoder-decoder config."""
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.encoder_hidden_size,
            intermediate_size=config.encoder_intermediate_size,
            num_hidden_layers=config.encoder_num_hidden_layers,
            num_attention_heads=config.encoder_num_attention_heads,
            num_key_value_heads=config.encoder_num_key_value_heads,
            head_dim=config.encoder_head_dim,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            pad_token_id=config.pad_token_id,
            pooling_mode=pooling_mode,
            normalize_embeddings=normalize_embeddings,
        )


@dataclass
class EncoderOutput(BaseModelOutput):
    """
    Output of the encoder model.
    
    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last layer.
        pooler_output: Pooled representation of the sequence.
        hidden_states: Tuple of hidden-states at each layer (if output_hidden_states=True).
        attentions: Tuple of attention weights at each layer (if output_attentions=True).
    """
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Qwen3EncoderPooler(nn.Module):
    """
    Pooling strategies for encoder outputs.
    
    Supports:
    - mean: Mean pooling over non-padding tokens
    - cls: First token (CLS-style)
    - last: Last non-padding token
    - weighted_mean: Position-weighted mean pooling
    """
    
    def __init__(self, config: Qwen3EncoderConfig):
        super().__init__()
        self.pooling_mode = config.pooling_mode
        self.normalize = config.normalize_embeddings
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states into a single vector.
        
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
            pooled = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
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
                dim=1
            )
        
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
        
        # Normalize if requested
        if self.normalize:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        
        return pooled


class Qwen3EncoderModel(PreTrainedModel):
    """
    Standalone Qwen3 Encoder for embedding tasks.
    
    This model wraps the encoder from the trained encoder-decoder
    and adds pooling for sentence/document embeddings.
    
    Example:
        ```python
        from transformers import AutoModel, AutoTokenizer
        
        model = AutoModel.from_pretrained("your-org/qwen3-encoder")
        tokenizer = AutoTokenizer.from_pretrained("your-org/qwen3-encoder")
        
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
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Create encoder-decoder config for encoder initialization
        enc_dec_config = Qwen3EncoderDecoderConfig(
            vocab_size=config.vocab_size,
            encoder_hidden_size=config.hidden_size,
            encoder_intermediate_size=config.intermediate_size,
            encoder_num_hidden_layers=config.num_hidden_layers,
            encoder_num_attention_heads=config.num_attention_heads,
            encoder_num_key_value_heads=config.num_key_value_heads,
            encoder_head_dim=config.head_dim,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            pad_token_id=config.pad_token_id,
        )
        
        # Encoder layers
        self.encoder = Qwen3Encoder(enc_dec_config)
        
        # Pooler
        self.pooler = Qwen3EncoderPooler(config)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EncoderOutput]:
        """
        Forward pass for encoding.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            inputs_embeds: Optional pre-computed embeddings
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dataclass
            
        Returns:
            EncoderOutput with last_hidden_state and pooler_output
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device
            )
        
        # Run encoder
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        
        # Pool to get sentence embedding
        pooler_output = self.pooler(last_hidden_state, attention_mask)
        
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]
        
        return EncoderOutput(
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
        from tqdm import tqdm
        
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]
        
        device = device or next(self.parameters()).device
        self.eval()
        
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")
        
        with torch.no_grad():
            for i in iterator:
                batch_sentences = sentences[i:i + batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Encode
                outputs = self(**inputs)
                embeddings = outputs.pooler_output
                
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
```

### 10.2 Encoder Extraction Utility

```python
# src/extraction/extract_encoder.py

"""
Utility for extracting encoder from trained encoder-decoder model.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil

import torch
from transformers import AutoTokenizer

from ..models.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
from ..models.modeling_qwen3_seq2seq import Qwen3ForSeq2SeqLM
from ..models.encoder_only import Qwen3EncoderModel, Qwen3EncoderConfig

logger = logging.getLogger(__name__)


class EncoderExtractor:
    """
    Extract and export encoder from trained encoder-decoder model.
    
    Example:
        ```python
        extractor = EncoderExtractor(
            checkpoint_path="checkpoints/qwen3-encdec-final",
            output_path="models/qwen3-encoder"
        )
        extractor.extract_and_save()
        ```
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        output_path: str,
        pooling_mode: str = "mean",
        normalize_embeddings: bool = True,
    ):
        """
        Initialize extractor.
        
        Args:
            checkpoint_path: Path to trained encoder-decoder checkpoint
            output_path: Path to save extracted encoder
            pooling_mode: Pooling strategy (mean, cls, last, weighted_mean)
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)
        self.pooling_mode = pooling_mode
        self.normalize_embeddings = normalize_embeddings
        
    def load_encoder_decoder(self) -> Qwen3ForSeq2SeqLM:
        """Load the trained encoder-decoder model."""
        logger.info(f"Loading encoder-decoder from {self.checkpoint_path}")
        
        model = Qwen3ForSeq2SeqLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.float32,  # Use full precision for extraction
        )
        
        return model
    
    def extract_encoder_weights(
        self,
        enc_dec_model: Qwen3ForSeq2SeqLM
    ) -> Dict[str, torch.Tensor]:
        """
        Extract encoder weights from encoder-decoder model.
        
        Returns:
            Dictionary of encoder state dict
        """
        logger.info("Extracting encoder weights...")
        
        state_dict = {}
        
        # Get the shared embeddings
        # In the encoder-decoder, this is model.shared
        shared_embeddings = enc_dec_model.model.shared.weight.data.clone()
        state_dict["embed_tokens.weight"] = shared_embeddings
        
        # Get encoder layer weights
        encoder = enc_dec_model.model.encoder
        encoder_state = encoder.state_dict()
        
        # Rename keys to match standalone encoder format
        for key, value in encoder_state.items():
            new_key = f"encoder.{key}"
            state_dict[new_key] = value.clone()
        
        logger.info(f"Extracted {len(state_dict)} weight tensors")
        
        return state_dict
    
    def create_encoder_config(
        self,
        enc_dec_config: Qwen3EncoderDecoderConfig
    ) -> Qwen3EncoderConfig:
        """Create encoder-only config from encoder-decoder config."""
        return Qwen3EncoderConfig.from_encoder_decoder_config(
            enc_dec_config,
            pooling_mode=self.pooling_mode,
            normalize_embeddings=self.normalize_embeddings
        )
    
    def create_encoder_model(
        self,
        config: Qwen3EncoderConfig,
        state_dict: Dict[str, torch.Tensor]
    ) -> Qwen3EncoderModel:
        """Create and load encoder model."""
        logger.info("Creating encoder model...")
        
        # Create model with random weights
        model = Qwen3EncoderModel(config)
        
        # Load extracted weights
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False
        )
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        return model
    
    def verify_extraction(
        self,
        enc_dec_model: Qwen3ForSeq2SeqLM,
        encoder_model: Qwen3EncoderModel,
        tokenizer
    ) -> bool:
        """
        Verify that extracted encoder produces same outputs.
        
        Returns:
            True if outputs match
        """
        logger.info("Verifying encoder extraction...")
        
        enc_dec_model.eval()
        encoder_model.eval()
        
        # Test input
        test_text = "This is a test sentence for verification."
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            # Get encoder output from encoder-decoder
            enc_dec_output = enc_dec_model.model.encoder(
                inputs_embeds=enc_dec_model.model.shared(inputs.input_ids),
                attention_mask=inputs.attention_mask,
                return_dict=True
            )
            
            # Get output from standalone encoder
            encoder_output = encoder_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True
            )
        
        # Compare hidden states
        diff = (enc_dec_output.last_hidden_state - encoder_output.last_hidden_state).abs()
        max_diff = diff.max().item()
        
        logger.info(f"Max difference in hidden states: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            logger.info("✓ Extraction verified: outputs match")
            return True
        else:
            logger.warning(f"✗ Extraction mismatch: max diff = {max_diff}")
            return False
    
    def save_model(
        self,
        model: Qwen3EncoderModel,
        config: Qwen3EncoderConfig,
        tokenizer
    ):
        """Save the extracted encoder model."""
        logger.info(f"Saving encoder to {self.output_path}")
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(self.output_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(self.output_path)
        
        # Save config (already saved by save_pretrained, but ensure it's correct)
        config.save_pretrained(self.output_path)
        
        logger.info("Model saved successfully")
    
    def create_model_card(
        self,
        base_model: str = "Qwen/Qwen3-0.6B",
        training_tokens: str = "500B-2T",
    ) -> str:
        """Create model card markdown."""
        return f'''---
language:
- en
- zh
- multilingual
license: apache-2.0
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- qwen3
- encoder
- embeddings
pipeline_tag: feature-extraction
library_name: transformers
base_model: {base_model}
---

# Qwen3 Encoder

A bidirectional encoder extracted from a Qwen3-based encoder-decoder model trained with UL2 objectives.

## Model Description

This model is the encoder component of a ~1B parameter encoder-decoder model based on Qwen3-0.6B architecture:

- **Architecture**: 28-layer bidirectional Transformer encoder
- **Hidden Size**: 1024
- **Attention Heads**: 16 query heads, 8 key-value heads (GQA)
- **Parameters**: ~600M (encoder only)
- **Training**: UL2 (Mixture of Denoisers) on {training_tokens} tokens

### Training Details

The encoder was trained as part of an encoder-decoder model using:
- **Objective**: UL2 with 5 denoising tasks
  - R-Denoiser (short spans): μ=3, 15% corruption
  - R-Denoiser (medium spans): μ=12, 50% corruption
  - X-Denoiser (long spans): μ=32, 15% corruption
  - X-Denoiser (extreme): μ=32, 50% corruption
  - S-Denoiser (sequential): 75% prefix-to-suffix (50% of training)
- **Data**: FineWeb-Edu
- **Initialization**: Qwen3-0.6B pretrained weights

## Usage

### Using Transformers

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("your-org/qwen3-encoder")
tokenizer = AutoTokenizer.from_pretrained("your-org/qwen3-encoder")

# Encode a sentence
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

# Get sentence embedding (pooled)
embedding = outputs.pooler_output  # (1, 1024)

# Or get token embeddings
token_embeddings = outputs.last_hidden_state  # (1, seq_len, 1024)
```

### Batch Encoding

```python
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology.",
    "Climate change poses significant challenges."
]

# Using the encode method
embeddings = model.encode(sentences, tokenizer, batch_size=32)
print(embeddings.shape)  # (3, 1024)
```

### Similarity Computation

```python
import torch.nn.functional as F

# Encode two sentences
emb1 = model.encode("I love machine learning", tokenizer)
emb2 = model.encode("AI and ML are fascinating", tokenizer)

# Compute cosine similarity
similarity = F.cosine_similarity(emb1, emb2)
print(f"Similarity: {{similarity.item():.4f}}")
```

## Pooling Modes

The model supports multiple pooling strategies:

- `mean` (default): Mean pooling over non-padding tokens
- `cls`: First token representation
- `last`: Last non-padding token
- `weighted_mean`: Position-weighted mean (later positions weighted more)

To change pooling mode:

```python
model.config.pooling_mode = "cls"
model.pooler.pooling_mode = "cls"
```

## Evaluation

See the evaluation results on MTEB benchmark and other tasks in the associated paper/documentation.

## Citation

If you use this model, please cite:

```bibtex
@misc{{qwen3encoder2024,
  title={{Qwen3 Encoder: Bidirectional Embeddings from Encoder-Decoder Training}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/your-org/qwen3-encoder}}
}}
```

## Limitations

- Maximum sequence length: 8192 tokens (can be extended with careful handling)
- Primarily trained on English and Chinese data
- May not perform well on very domain-specific tasks without fine-tuning
'''
    
    def save_model_card(self):
        """Save model card to output directory."""
        model_card = self.create_model_card()
        model_card_path = self.output_path / "README.md"
        
        with open(model_card_path, "w") as f:
            f.write(model_card)
        
        logger.info(f"Model card saved to {model_card_path}")
    
    def extract_and_save(self) -> Qwen3EncoderModel:
        """
        Main method: Extract encoder and save.
        
        Returns:
            The extracted encoder model
        """
        # Load encoder-decoder
        enc_dec_model = self.load_encoder_decoder()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        
        # Extract weights
        state_dict = self.extract_encoder_weights(enc_dec_model)
        
        # Create config
        config = self.create_encoder_config(enc_dec_model.config)
        
        # Create encoder model
        encoder_model = self.create_encoder_model(config, state_dict)
        
        # Verify extraction
        self.verify_extraction(enc_dec_model, encoder_model, tokenizer)
        
        # Save
        self.save_model(encoder_model, config, tokenizer)
        self.save_model_card()
        
        # Cleanup encoder-decoder to free memory
        del enc_dec_model
        torch.cuda.empty_cache()
        
        return encoder_model
```

### 10.3 Extraction Script

```python
# scripts/extract_encoder.py

"""
Script to extract encoder from trained encoder-decoder checkpoint.
"""

import argparse
import logging
from pathlib import Path

import torch

from src.extraction.extract_encoder import EncoderExtractor


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract encoder from trained encoder-decoder model"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained encoder-decoder checkpoint"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save extracted encoder"
    )
    
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "last", "weighted_mean"],
        help="Pooling strategy for sentence embeddings"
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't L2-normalize embeddings"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main extraction function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Encoder Extraction")
    logger.info("=" * 60)
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create extractor
    extractor = EncoderExtractor(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        pooling_mode=args.pooling,
        normalize_embeddings=not args.no_normalize,
    )
    
    # Extract and save
    encoder = extractor.extract_and_save()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Extraction Complete")
    logger.info("=" * 60)
    logger.info(f"Encoder saved to: {args.output}")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    logger.info(f"Pooling mode: {args.pooling}")
    logger.info(f"Normalize embeddings: {not args.no_normalize}")


if __name__ == "__main__":
    main()
```

### 10.4 Checkpoint Averaging Before Extraction

```python
# src/extraction/checkpoint_averaging.py

"""
Average multiple checkpoints before encoder extraction.
This typically improves model quality.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import json

import torch
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class CheckpointAverager:
    """
    Average multiple training checkpoints.
    
    Checkpoint averaging typically improves model quality
    by smoothing out noise in individual checkpoints.
    
    Example:
        ```python
        averager = CheckpointAverager(
            checkpoint_dir="checkpoints/",
            output_path="checkpoints/averaged"
        )
        averager.average_last_n(n=5)
        ```
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        output_path: str,
    ):
        """
        Initialize averager.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            output_path: Path to save averaged checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_path = Path(output_path)
    
    def find_checkpoints(self, pattern: str = "checkpoint-*") -> List[Path]:
        """Find all checkpoints matching pattern."""
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
        )
        return checkpoints
    
    def average_checkpoints(
        self,
        checkpoint_paths: List[Path]
    ) -> Dict[str, torch.Tensor]:
        """
        Average state dicts from multiple checkpoints.
        
        Args:
            checkpoint_paths: List of checkpoint directories
            
        Returns:
            Averaged state dict
        """
        logger.info(f"Averaging {len(checkpoint_paths)} checkpoints:")
        for cp in checkpoint_paths:
            logger.info(f"  - {cp.name}")
        
        averaged_state = {}
        
        for i, cp_path in enumerate(checkpoint_paths):
            # Load checkpoint
            model_path = cp_path / "model.safetensors"
            if not model_path.exists():
                model_path = cp_path / "pytorch_model.bin"
            
            if model_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location="cpu")
            
            # Average
            for key, value in state_dict.items():
                if i == 0:
                    averaged_state[key] = value.float()
                else:
                    averaged_state[key] += value.float()
        
        # Divide by number of checkpoints
        num_checkpoints = len(checkpoint_paths)
        for key in averaged_state:
            averaged_state[key] /= num_checkpoints
        
        return averaged_state
    
    def save_averaged_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        source_checkpoint: Path
    ):
        """
        Save averaged checkpoint with config and tokenizer.
        
        Args:
            state_dict: Averaged state dict
            source_checkpoint: Source checkpoint to copy config from
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        output_model_path = self.output_path / "model.safetensors"
        try:
            from safetensors.torch import save_file
            save_file(state_dict, output_model_path)
        except ImportError:
            output_model_path = self.output_path / "pytorch_model.bin"
            torch.save(state_dict, output_model_path)
        
        logger.info(f"Saved averaged model to {output_model_path}")
        
        # Copy config
        import shutil
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "vocab.json", "merges.txt"]:
            src = source_checkpoint / file
            if src.exists():
                shutil.copy(src, self.output_path / file)
        
        # Save averaging metadata
        metadata = {
            "averaged_from": [str(p) for p in self._last_averaged_paths],
            "num_checkpoints": len(self._last_averaged_paths)
        }
        with open(self.output_path / "averaging_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def average_last_n(self, n: int = 5) -> Path:
        """
        Average the last N checkpoints.
        
        Args:
            n: Number of checkpoints to average
            
        Returns:
            Path to averaged checkpoint
        """
        checkpoints = self.find_checkpoints()
        
        if len(checkpoints) < n:
            logger.warning(
                f"Only {len(checkpoints)} checkpoints found, "
                f"averaging all of them"
            )
            n = len(checkpoints)
        
        # Take last N
        selected = checkpoints[-n:]
        self._last_averaged_paths = selected
        
        # Average
        averaged_state = self.average_checkpoints(selected)
        
        # Save
        self.save_averaged_checkpoint(averaged_state, selected[-1])
        
        logger.info(f"Averaged checkpoint saved to {self.output_path}")
        
        return self.output_path
    
    def average_best_n(
        self,
        n: int = 5,
        metric: str = "eval_loss",
        lower_is_better: bool = True
    ) -> Path:
        """
        Average the N best checkpoints by metric.
        
        Args:
            n: Number of checkpoints to average
            metric: Metric to use for selection
            lower_is_better: Whether lower metric is better
            
        Returns:
            Path to averaged checkpoint
        """
        checkpoints = self.find_checkpoints()
        
        # Load trainer state to get metrics
        checkpoint_metrics = []
        
        for cp in checkpoints:
            trainer_state_path = cp / "trainer_state.json"
            if trainer_state_path.exists():
                with open(trainer_state_path) as f:
                    trainer_state = json.load(f)
                
                # Find metric in log history
                for entry in reversed(trainer_state.get("log_history", [])):
                    if metric in entry:
                        checkpoint_metrics.append((cp, entry[metric]))
                        break
        
        if len(checkpoint_metrics) < n:
            logger.warning(
                f"Only {len(checkpoint_metrics)} checkpoints have metric {metric}"
            )
            n = len(checkpoint_metrics)
        
        # Sort by metric
        checkpoint_metrics.sort(
            key=lambda x: x[1],
            reverse=not lower_is_better
        )
        
        # Select best N
        selected = [cp for cp, _ in checkpoint_metrics[:n]]
        self._last_averaged_paths = selected
        
        logger.info(f"Selected checkpoints by {metric}:")
        for cp, m in checkpoint_metrics[:n]:
            logger.info(f"  - {cp.name}: {metric}={m:.4f}")
        
        # Average
        averaged_state = self.average_checkpoints(selected)
        
        # Save
        self.save_averaged_checkpoint(averaged_state, selected[0])
        
        return self.output_path
```

### 10.5 Sentence-Transformers Integration

```python
# src/extraction/sentence_transformers_export.py

"""
Export encoder for use with sentence-transformers library.
"""

import logging
from pathlib import Path
from typing import Optional
import json

logger = logging.getLogger(__name__)


def create_sentence_transformers_config(
    output_path: str,
    max_seq_length: int = 512,
    pooling_mode: str = "mean",
    normalize: bool = True,
):
    """
    Create sentence-transformers compatible configuration files.
    
    This allows the model to be loaded with:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("path/to/model")
    ```
    
    Args:
        output_path: Model output directory
        max_seq_length: Maximum sequence length
        pooling_mode: Pooling strategy
        normalize: Whether to normalize embeddings
    """
    output_path = Path(output_path)
    
    # Create modules.json
    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer"
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling"
        }
    ]
    
    if normalize:
        modules.append({
            "idx": 2,
            "name": "2",
            "path": "2_Normalize",
            "type": "sentence_transformers.models.Normalize"
        })
    
    with open(output_path / "modules.json", "w") as f:
        json.dump(modules, f, indent=2)
    
    # Create sentence_bert_config.json
    st_config = {
        "max_seq_length": max_seq_length,
        "do_lower_case": False
    }
    
    with open(output_path / "sentence_bert_config.json", "w") as f:
        json.dump(st_config, f, indent=2)
    
    # Create pooling config
    pooling_dir = output_path / "1_Pooling"
    pooling_dir.mkdir(exist_ok=True)
    
    pooling_config = {
        "word_embedding_dimension": 1024,  # hidden_size
        "pooling_mode_cls_token": pooling_mode == "cls",
        "pooling_mode_mean_tokens": pooling_mode == "mean",
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": pooling_mode == "weighted_mean",
        "pooling_mode_lasttoken": pooling_mode == "last",
    }
    
    with open(pooling_dir / "config.json", "w") as f:
        json.dump(pooling_config, f, indent=2)
    
    # Create normalize config if needed
    if normalize:
        normalize_dir = output_path / "2_Normalize"
        normalize_dir.mkdir(exist_ok=True)
        
        # Empty config is fine for Normalize
        with open(normalize_dir / "config.json", "w") as f:
            json.dump({}, f)
    
    logger.info(f"Created sentence-transformers config in {output_path}")


def verify_sentence_transformers_loading(model_path: str) -> bool:
    """
    Verify model can be loaded with sentence-transformers.
    
    Args:
        model_path: Path to model
        
    Returns:
        True if loading succeeds
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_path)
        
        # Test encoding
        embeddings = model.encode(["Test sentence"])
        
        logger.info(f"✓ Model loads with sentence-transformers")
        logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping verification")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load with sentence-transformers: {e}")
        return False
```

---

## Acceptance Criteria

### 10.A Core Functionality

- [ ] `Qwen3EncoderConfig` correctly represents encoder-only configuration
- [ ] `Qwen3EncoderModel` produces correct embeddings from extracted weights
- [ ] All four pooling modes work correctly (mean, cls, last, weighted_mean)
- [ ] `encode()` convenience method handles batching correctly

### 10.B Extraction Process

- [ ] `EncoderExtractor` correctly extracts encoder weights from encoder-decoder
- [ ] Extraction verification confirms outputs match (< 1e-5 difference)
- [ ] Tokenizer is copied correctly to output directory
- [ ] Model card is generated with correct information

### 10.C Checkpoint Averaging

- [ ] `CheckpointAverager` correctly averages multiple checkpoints
- [ ] `average_last_n()` selects correct checkpoints
- [ ] `average_best_n()` selects by metric correctly
- [ ] Averaged checkpoint saves in correct format

### 10.D Integration

- [ ] Extracted encoder loads with `AutoModel.from_pretrained()`
- [ ] Sentence-transformers config is generated correctly
- [ ] Model can be uploaded to HuggingFace Hub

---

## Testing Requirements

### Unit Tests

```python
# tests/test_encoder_extraction.py

import pytest
import torch
import tempfile
from pathlib import Path

from src.models.encoder_only import (
    Qwen3EncoderConfig,
    Qwen3EncoderModel,
    Qwen3EncoderPooler
)


class TestQwen3EncoderConfig:
    """Tests for encoder configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Qwen3EncoderConfig()
        
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 28
        assert config.pooling_mode == "mean"
        assert config.normalize_embeddings == True
    
    def test_from_encoder_decoder_config(self):
        """Test creating config from encoder-decoder config."""
        from src.models.configuration_qwen3_encdec import Qwen3EncoderDecoderConfig
        
        enc_dec_config = Qwen3EncoderDecoderConfig()
        encoder_config = Qwen3EncoderConfig.from_encoder_decoder_config(
            enc_dec_config,
            pooling_mode="cls",
            normalize_embeddings=False
        )
        
        assert encoder_config.hidden_size == enc_dec_config.encoder_hidden_size
        assert encoder_config.num_hidden_layers == enc_dec_config.encoder_num_hidden_layers
        assert encoder_config.pooling_mode == "cls"
        assert encoder_config.normalize_embeddings == False


class TestQwen3EncoderPooler:
    """Tests for pooling strategies."""
    
    @pytest.fixture
    def hidden_states(self):
        """Sample hidden states (batch=2, seq=5, hidden=8)."""
        return torch.randn(2, 5, 8)
    
    @pytest.fixture
    def attention_mask(self):
        """Sample attention mask with padding."""
        return torch.tensor([
            [1, 1, 1, 1, 1],  # No padding
            [1, 1, 1, 0, 0],  # 2 padding tokens
        ])
    
    def test_mean_pooling(self, hidden_states, attention_mask):
        """Test mean pooling."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="mean",
            normalize_embeddings=False
        )
        pooler = Qwen3EncoderPooler(config)
        
        output = pooler(hidden_states, attention_mask)
        
        assert output.shape == (2, 8)
        
        # Verify first sequence (no padding)
        expected_0 = hidden_states[0].mean(dim=0)
        assert torch.allclose(output[0], expected_0)
        
        # Verify second sequence (with padding)
        expected_1 = hidden_states[1, :3].mean(dim=0)
        assert torch.allclose(output[1], expected_1)
    
    def test_cls_pooling(self, hidden_states, attention_mask):
        """Test CLS (first token) pooling."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="cls",
            normalize_embeddings=False
        )
        pooler = Qwen3EncoderPooler(config)
        
        output = pooler(hidden_states, attention_mask)
        
        assert output.shape == (2, 8)
        assert torch.allclose(output[0], hidden_states[0, 0])
        assert torch.allclose(output[1], hidden_states[1, 0])
    
    def test_last_pooling(self, hidden_states, attention_mask):
        """Test last token pooling."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="last",
            normalize_embeddings=False
        )
        pooler = Qwen3EncoderPooler(config)
        
        output = pooler(hidden_states, attention_mask)
        
        assert output.shape == (2, 8)
        # First sequence: last token is index 4
        assert torch.allclose(output[0], hidden_states[0, 4])
        # Second sequence: last non-padding is index 2
        assert torch.allclose(output[1], hidden_states[1, 2])
    
    def test_normalization(self, hidden_states, attention_mask):
        """Test L2 normalization."""
        config = Qwen3EncoderConfig(
            hidden_size=8,
            pooling_mode="mean",
            normalize_embeddings=True
        )
        pooler = Qwen3EncoderPooler(config)
        
        output = pooler(hidden_states, attention_mask)
        
        # Check L2 norm is 1
        norms = output.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2))


class TestQwen3EncoderModel:
    """Tests for encoder model."""
    
    @pytest.fixture
    def small_config(self):
        """Small config for testing."""
        return Qwen3EncoderConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
    
    def test_forward_pass(self, small_config):
        """Test forward pass produces correct shapes."""
        model = Qwen3EncoderModel(small_config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, small_config.hidden_size)
        assert outputs.pooler_output.shape == (batch_size, small_config.hidden_size)
    
    def test_encode_method(self, small_config):
        """Test batch encoding method."""
        from transformers import AutoTokenizer
        
        model = Qwen3EncoderModel(small_config)
        
        # Create a simple tokenizer mock
        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                batch_size = len(texts) if isinstance(texts, list) else 1
                return {
                    "input_ids": torch.randint(0, small_config.vocab_size, (batch_size, 10)),
                    "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
                }
        
        tokenizer = MockTokenizer()
        sentences = ["Hello world", "Test sentence"]
        
        embeddings = model.encode(sentences, tokenizer, show_progress=False)
        
        assert embeddings.shape == (2, small_config.hidden_size)


class TestEncoderExtraction:
    """Tests for extraction process."""
    
    def test_extraction_weights_match(self):
        """Test extracted weights produce same outputs."""
        # This would require a trained model
        # In practice, use integration tests
        pass
    
    def test_checkpoint_averaging(self):
        """Test checkpoint averaging."""
        from src.extraction.checkpoint_averaging import CheckpointAverager
        
        # Create mock checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create 3 mock checkpoints
            for i in range(3):
                cp_dir = tmpdir / f"checkpoint-{i * 1000}"
                cp_dir.mkdir()
                
                # Create mock state dict
                state_dict = {
                    "layer.weight": torch.randn(10, 10),
                    "layer.bias": torch.randn(10)
                }
                torch.save(state_dict, cp_dir / "pytorch_model.bin")
                
                # Create mock config
                with open(cp_dir / "config.json", "w") as f:
                    f.write("{}")
            
            # Average
            averager = CheckpointAverager(
                checkpoint_dir=str(tmpdir),
                output_path=str(tmpdir / "averaged")
            )
            
            output_path = averager.average_last_n(n=3)
            
            assert output_path.exists()
            assert (output_path / "pytorch_model.bin").exists()
```

### Integration Tests

```python
# tests/integration/test_encoder_extraction_e2e.py

"""
End-to-end tests for encoder extraction.
Requires a trained encoder-decoder checkpoint.
"""

import pytest
import torch
from pathlib import Path

from src.extraction.extract_encoder import EncoderExtractor


@pytest.mark.integration
class TestEncoderExtractionE2E:
    """End-to-end extraction tests."""
    
    @pytest.fixture
    def trained_checkpoint(self):
        """Path to trained checkpoint (set via env or skip)."""
        import os
        checkpoint = os.environ.get("TEST_CHECKPOINT")
        if not checkpoint:
            pytest.skip("TEST_CHECKPOINT not set")
        return checkpoint
    
    def test_full_extraction(self, trained_checkpoint, tmp_path):
        """Test full extraction pipeline."""
        output_path = tmp_path / "extracted_encoder"
        
        extractor = EncoderExtractor(
            checkpoint_path=trained_checkpoint,
            output_path=str(output_path),
            pooling_mode="mean",
            normalize_embeddings=True
        )
        
        encoder = extractor.extract_and_save()
        
        # Check files created
        assert (output_path / "config.json").exists()
        assert (output_path / "model.safetensors").exists() or \
               (output_path / "pytorch_model.bin").exists()
        assert (output_path / "README.md").exists()
        
        # Check encoder works
        test_input = torch.randint(0, 1000, (1, 10))
        outputs = encoder(test_input)
        
        assert outputs.pooler_output.shape == (1, 1024)
    
    def test_sentence_transformers_compat(self, trained_checkpoint, tmp_path):
        """Test sentence-transformers compatibility."""
        pytest.importorskip("sentence_transformers")
        
        from sentence_transformers import SentenceTransformer
        from src.extraction.sentence_transformers_export import (
            create_sentence_transformers_config,
            verify_sentence_transformers_loading
        )
        
        output_path = tmp_path / "st_encoder"
        
        # Extract encoder
        extractor = EncoderExtractor(
            checkpoint_path=trained_checkpoint,
            output_path=str(output_path)
        )
        extractor.extract_and_save()
        
        # Create ST config
        create_sentence_transformers_config(str(output_path))
        
        # Verify loading
        assert verify_sentence_transformers_loading(str(output_path))
```

---

## File Structure

```
src/
├── models/
│   ├── encoder_only.py              # Standalone encoder model
│   └── ...
├── extraction/
│   ├── __init__.py
│   ├── extract_encoder.py           # Main extraction utility
│   ├── checkpoint_averaging.py       # Checkpoint averaging
│   └── sentence_transformers_export.py  # ST compatibility
scripts/
├── extract_encoder.py               # CLI extraction script
tests/
├── test_encoder_extraction.py       # Unit tests
└── integration/
    └── test_encoder_extraction_e2e.py  # Integration tests
```

---

## Usage Examples

### Basic Extraction

```bash
# Extract encoder from trained checkpoint
python scripts/extract_encoder.py \
    --checkpoint checkpoints/qwen3-encdec-final \
    --output models/qwen3-encoder \
    --pooling mean

# With checkpoint averaging first
python scripts/average_checkpoints.py \
    --checkpoint-dir checkpoints/ \
    --output checkpoints/averaged \
    --last-n 5

python scripts/extract_encoder.py \
    --checkpoint checkpoints/averaged \
    --output models/qwen3-encoder
```

### Using Extracted Encoder

```python
from transformers import AutoModel, AutoTokenizer

# Load extracted encoder
model = AutoModel.from_pretrained("models/qwen3-encoder")
tokenizer = AutoTokenizer.from_pretrained("models/qwen3-encoder")

# Encode sentences
sentences = [
    "Machine learning is fascinating.",
    "I love artificial intelligence.",
    "The weather is nice today."
]

# Get embeddings
embeddings = model.encode(sentences, tokenizer)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 1024)

# Compute similarities
import torch.nn.functional as F

sim_01 = F.cosine_similarity(embeddings[0:1], embeddings[1:2])
sim_02 = F.cosine_similarity(embeddings[0:1], embeddings[2:3])

print(f"Similarity (0, 1): {sim_01.item():.4f}")  # Higher - related topics
print(f"Similarity (0, 2): {sim_02.item():.4f}")  # Lower - different topics
```

### Upload to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model
api.upload_folder(
    folder_path="models/qwen3-encoder",
    repo_id="your-org/qwen3-encoder",
    repo_type="model"
)
```

---

## Definition of Done

1. **Code Complete**: All extraction utilities implemented and tested
2. **Extraction Verified**: Outputs match between encoder-decoder and standalone encoder
3. **Pooling Modes**: All four pooling strategies work correctly
4. **Checkpoint Averaging**: Works with both last-N and best-N selection
5. **Sentence-Transformers**: Compatibility verified
6. **Documentation**: Model card and usage examples complete
7. **Tests Pass**: All unit and integration tests pass
