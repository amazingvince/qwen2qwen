"""Average multiple checkpoints before encoder extraction.

Checkpoint averaging typically improves model quality by smoothing out
noise in individual checkpoints.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch

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
        self._last_averaged_paths: List[Path] = []

    def find_checkpoints(self, pattern: str = "checkpoint-*") -> List[Path]:
        """Find all checkpoints matching pattern."""
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: (
                int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
            ),
        )
        return checkpoints

    def average_checkpoints(
        self,
        checkpoint_paths: List[Path],
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

        averaged_state: Dict[str, torch.Tensor] = {}

        for i, cp_path in enumerate(checkpoint_paths):
            # Load checkpoint
            model_path = cp_path / "model.safetensors"
            if not model_path.exists():
                model_path = cp_path / "pytorch_model.bin"

            if not model_path.exists():
                raise FileNotFoundError(f"No model file found in {cp_path}")

            if model_path.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file

                    state_dict = load_file(model_path)
                except ImportError:
                    raise ImportError(
                        "safetensors is required for .safetensors files. "
                        "Install with: pip install safetensors"
                    )
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
        source_checkpoint: Path,
    ) -> None:
        """
        Save averaged checkpoint with config and tokenizer.

        Args:
            state_dict: Averaged state dict
            source_checkpoint: Source checkpoint to copy config from
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        try:
            from safetensors.torch import save_file

            output_model_path = self.output_path / "model.safetensors"
            save_file(state_dict, output_model_path)
        except ImportError:
            output_model_path = self.output_path / "pytorch_model.bin"
            torch.save(state_dict, output_model_path)

        logger.info(f"Saved averaged model to {output_model_path}")

        # Copy config and tokenizer files
        files_to_copy = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
        ]

        for file in files_to_copy:
            src = source_checkpoint / file
            if src.exists():
                shutil.copy(src, self.output_path / file)

        # Save averaging metadata
        metadata = {
            "averaged_from": [str(p) for p in self._last_averaged_paths],
            "num_checkpoints": len(self._last_averaged_paths),
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

        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")

        if len(checkpoints) < n:
            logger.warning(
                f"Only {len(checkpoints)} checkpoints found, averaging all of them"
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
        lower_is_better: bool = True,
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

        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")

        # Load trainer state to get metrics
        checkpoint_metrics: List[tuple] = []

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

        if len(checkpoint_metrics) == 0:
            raise ValueError(
                f"No checkpoints found with metric {metric} in trainer_state.json"
            )

        if len(checkpoint_metrics) < n:
            logger.warning(
                f"Only {len(checkpoint_metrics)} checkpoints have metric {metric}"
            )
            n = len(checkpoint_metrics)

        # Sort by metric
        checkpoint_metrics.sort(
            key=lambda x: x[1],
            reverse=not lower_is_better,
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
