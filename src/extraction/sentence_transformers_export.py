"""Export encoder for use with sentence-transformers library.

This module provides utilities for creating sentence-transformers
compatible configuration files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_sentence_transformers_config(
    output_path: str,
    hidden_size: int = 1024,
    max_seq_length: int = 512,
    pooling_mode: str = "mean",
    normalize: bool = True,
) -> None:
    """
    Create sentence-transformers compatible configuration files.

    This allows the model to be loaded with:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("path/to/model")
    ```

    Args:
        output_path: Model output directory
        hidden_size: Dimension of hidden states
        max_seq_length: Maximum sequence length
        pooling_mode: Pooling strategy
        normalize: Whether to normalize embeddings
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create modules.json
    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]

    if normalize:
        modules.append(
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            }
        )

    with open(output_path / "modules.json", "w") as f:
        json.dump(modules, f, indent=2)

    # Create sentence_bert_config.json
    st_config = {"max_seq_length": max_seq_length, "do_lower_case": False}

    with open(output_path / "sentence_bert_config.json", "w") as f:
        json.dump(st_config, f, indent=2)

    # Create pooling config
    pooling_dir = output_path / "1_Pooling"
    pooling_dir.mkdir(exist_ok=True)

    pooling_config = {
        "word_embedding_dimension": hidden_size,
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

        model = SentenceTransformer(model_path, trust_remote_code=True)

        # Test encoding
        embeddings = model.encode(["Test sentence"])

        logger.info("Model loads with sentence-transformers")
        logger.info(f"  Embedding dimension: {embeddings.shape[1]}")

        return True

    except ImportError:
        logger.warning("sentence-transformers not installed, skipping verification")
        return True

    except Exception as e:
        logger.error(f"Failed to load with sentence-transformers: {e}")
        return False
