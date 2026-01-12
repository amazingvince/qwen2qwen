"""Utility for extracting encoder from trained encoder-decoder model.

This module provides the EncoderExtractor class which handles the full
extraction pipeline from a trained encoder-decoder checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoTokenizer

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

    def load_encoder_decoder(self) -> Any:
        """Load the trained encoder-decoder model."""
        # Import here to avoid circular imports
        from qwen3_encdec import Qwen3ForSeq2SeqLM

        logger.info(f"Loading encoder-decoder from {self.checkpoint_path}")

        model = Qwen3ForSeq2SeqLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.float32,  # Use full precision for extraction
        )

        return model

    def extract_encoder_weights(
        self,
        enc_dec_model: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract encoder weights from encoder-decoder model.

        Returns:
            Dictionary of encoder state dict with corrected key names.
        """
        logger.info("Extracting encoder weights...")

        # Get the encoder state dict directly
        encoder = enc_dec_model.model.encoder
        state_dict = encoder.state_dict()

        # Create new state dict with encoder. prefix for standalone model
        extracted_state = {}
        for key, value in state_dict.items():
            new_key = f"encoder.{key}"
            extracted_state[new_key] = value.clone()

        logger.info(f"Extracted {len(extracted_state)} weight tensors")

        return extracted_state

    def create_encoder_config(self, enc_dec_config: Any) -> Any:
        """Create encoder-only config from encoder-decoder config."""
        from qwen3_encdec.encoder_only import Qwen3EncoderConfig

        return Qwen3EncoderConfig.from_encoder_decoder_config(
            enc_dec_config,
            pooling_mode=self.pooling_mode,
            normalize_embeddings=self.normalize_embeddings,
        )

    def create_encoder_model(
        self,
        config: Any,
        state_dict: Dict[str, torch.Tensor],
    ) -> Any:
        """Create and load encoder model."""
        from qwen3_encdec.encoder_only import Qwen3StandaloneEncoderModel

        logger.info("Creating encoder model...")

        # Create model with random weights
        model = Qwen3StandaloneEncoderModel(config)

        # Load extracted weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        return model

    def verify_extraction(
        self,
        enc_dec_model: Any,
        encoder_model: Any,
        tokenizer: Any,
    ) -> bool:
        """
        Verify that extracted encoder produces same outputs.

        Returns:
            True if outputs match (within tolerance)
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
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )

            # Get output from standalone encoder
            encoder_output = encoder_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )

        # Compare hidden states
        diff = (
            enc_dec_output.last_hidden_state - encoder_output.last_hidden_state
        ).abs()
        max_diff = diff.max().item()

        logger.info(f"Max difference in hidden states: {max_diff:.2e}")

        if max_diff < 1e-5:
            logger.info("Extraction verified: outputs match")
            return True
        else:
            logger.warning(f"Extraction mismatch: max diff = {max_diff}")
            return False

    def save_model(
        self,
        model: Any,
        config: Any,
        tokenizer: Any,
    ) -> None:
        """Save the extracted encoder model."""
        logger.info(f"Saving encoder to {self.output_path}")

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_pretrained(self.output_path)

        # Save tokenizer
        tokenizer.save_pretrained(self.output_path)

        logger.info("Model saved successfully")

    def create_model_card(
        self,
        base_model: str = "Qwen/Qwen3-0.6B",
        training_tokens: str = "500B-2T",
    ) -> str:
        """Create model card markdown."""
        return f"""---
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
  - R-Denoiser (short spans): mean=3, 15% corruption
  - R-Denoiser (medium spans): mean=12, 50% corruption
  - X-Denoiser (long spans): mean=32, 15% corruption
  - X-Denoiser (extreme): mean=32, 50% corruption
  - S-Denoiser (sequential): 75% prefix-to-suffix (50% of training)
- **Data**: FineWeb-Edu
- **Initialization**: Qwen3-0.6B pretrained weights

## Usage

### Using Transformers

```python
from transformers import AutoTokenizer
from qwen3_encdec import Qwen3StandaloneEncoderModel

model = Qwen3StandaloneEncoderModel.from_pretrained("your-org/qwen3-encoder")
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
"""

    def save_model_card(self) -> None:
        """Save model card to output directory."""
        model_card = self.create_model_card()
        model_card_path = self.output_path / "README.md"

        with open(model_card_path, "w") as f:
            f.write(model_card)

        logger.info(f"Model card saved to {model_card_path}")

    def extract_and_save(self) -> Any:
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return encoder_model
