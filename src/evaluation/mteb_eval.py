"""MTEB (Massive Text Embedding Benchmark) evaluation for Qwen3 Encoder.

MTEB evaluates embeddings across 8 task categories:
- Classification
- Clustering
- Pair Classification
- Reranking
- Retrieval
- STS (Semantic Textual Similarity)
- Summarization
- BitextMining
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class MTEBConfig:
    """Configuration for MTEB evaluation."""

    # Model settings
    model_path: str = "./extracted_encoder"
    model_name: str = "qwen3-encoder"

    # Evaluation settings
    task_categories: List[str] = field(
        default_factory=lambda: [
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "STS",
            "Summarization",
        ]
    )

    # Specific tasks (if None, run all in categories)
    task_names: Optional[List[str]] = None

    # Language filter
    languages: List[str] = field(default_factory=lambda: ["eng"])

    # Evaluation parameters
    batch_size: int = 32
    max_seq_length: int = 512

    # Output settings
    output_dir: str = "./mteb_results"
    save_predictions: bool = False

    # Hardware
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class Qwen3EncoderWrapper:
    """
    Wrapper to make Qwen3Encoder compatible with MTEB evaluation.

    MTEB expects a model with an `encode` method that takes:
    - sentences: List[str]
    - batch_size: int
    - show_progress_bar: bool

    And returns: np.ndarray of shape (num_sentences, embedding_dim)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
        pooling_mode: str = "mean",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.pooling_mode = pooling_mode

        # Load model and tokenizer
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the Qwen3 encoder model."""
        from transformers import AutoTokenizer

        # Try loading as sentence-transformers model first
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_path, device=self.device)
            self.use_sentence_transformers = True
            logger.info(f"Loaded model via SentenceTransformers from {model_path}")
        except Exception as e:
            logger.info(
                f"SentenceTransformers loading failed, trying direct load: {e}"
            )

            # Fall back to direct loading
            from qwen3_encdec.encoder_only import Qwen3StandaloneEncoderModel

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = Qwen3StandaloneEncoderModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.use_sentence_transformers = False
            logger.info(f"Loaded model directly from {model_path}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: Optional[bool] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings.

        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            normalize_embeddings: Whether to L2 normalize embeddings

        Returns:
            Embeddings as numpy array or torch tensor
        """
        if normalize_embeddings is None:
            normalize_embeddings = self.normalize_embeddings

        if self.use_sentence_transformers:
            return self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )

        # Manual encoding
        all_embeddings = []

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(
                iterator, desc="Encoding", total=len(sentences) // batch_size + 1
            )

        with torch.no_grad():
            for start_idx in iterator:
                batch = sentences[start_idx : start_idx + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output  # Shape: (batch_size, hidden_size)

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings


class MTEBEvaluator:
    """MTEB evaluation runner for Qwen3 Encoder."""

    def __init__(self, config: MTEBConfig):
        self.config = config
        self.results: Dict[str, Any] = {}

        # Initialize model wrapper
        self.model = Qwen3EncoderWrapper(
            model_path=config.model_path,
            device=config.device,
            max_seq_length=config.max_seq_length,
        )

    def get_tasks(self) -> List[Any]:
        """Get MTEB tasks to evaluate on."""
        try:
            from mteb import get_tasks
        except ImportError:
            raise ImportError(
                "mteb is required for MTEB evaluation. Install with: pip install mteb"
            )

        if self.config.task_names:
            # Specific tasks requested
            tasks = get_tasks(tasks=self.config.task_names)
        else:
            # Get all tasks in specified categories
            tasks = get_tasks(
                task_types=self.config.task_categories,
                languages=self.config.languages,
            )

        logger.info(f"Loaded {len(tasks)} tasks for evaluation")
        return tasks

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full MTEB evaluation.

        Returns:
            Dictionary of results per task and category
        """
        try:
            from mteb import MTEB
        except ImportError:
            raise ImportError(
                "mteb is required for MTEB evaluation. Install with: pip install mteb"
            )

        tasks = self.get_tasks()

        # Initialize MTEB
        evaluation = MTEB(tasks=tasks)

        # Run evaluation
        logger.info("Starting MTEB evaluation...")
        results = evaluation.run(
            self.model,
            output_folder=self.config.output_dir,
            eval_splits=["test"],
            batch_size=self.config.batch_size,
        )

        self.results = results
        return results

    def run_single_task(self, task_name: str) -> Dict[str, Any]:
        """
        Run evaluation on a single task.

        Args:
            task_name: Name of the MTEB task

        Returns:
            Task results dictionary
        """
        try:
            from mteb import MTEB, get_tasks
        except ImportError:
            raise ImportError(
                "mteb is required for MTEB evaluation. Install with: pip install mteb"
            )

        tasks = get_tasks(tasks=[task_name])

        evaluation = MTEB(tasks=tasks)
        results = evaluation.run(
            self.model,
            output_folder=self.config.output_dir,
            eval_splits=["test"],
            batch_size=self.config.batch_size,
        )

        return results

    def summarize_results(self) -> Dict[str, float]:
        """
        Summarize results by category.

        Returns:
            Dictionary with average scores per category
        """
        if not self.results:
            raise ValueError("No results to summarize. Run evaluation first.")

        category_scores: Dict[str, List[float]] = {}

        for task_name, task_results in self.results.items():
            # Get the main score (usually the first metric)
            if "test" in task_results:
                scores = task_results["test"]
                main_score = scores.get("main_score", list(scores.values())[0])

                # Determine category from task
                category = self._get_task_category(task_name)

                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(main_score)

        # Average per category
        summary = {}
        for category, scores in category_scores.items():
            summary[category] = float(np.mean(scores))

        # Overall average
        all_scores = [s for scores in category_scores.values() for s in scores]
        summary["Overall"] = float(np.mean(all_scores))

        return summary

    def _get_task_category(self, task_name: str) -> str:
        """Determine task category from task name."""
        category_keywords = {
            "Classification": ["class", "sentiment", "emotion", "toxic"],
            "Clustering": ["cluster"],
            "PairClassification": ["pair", "nli", "wnli"],
            "Reranking": ["rerank"],
            "Retrieval": ["retrieval", "msmarco", "nfcorpus", "scifact"],
            "STS": ["sts", "sick", "biosses"],
            "Summarization": ["summ"],
        }

        task_lower = task_name.lower()
        for category, keywords in category_keywords.items():
            if any(kw in task_lower for kw in keywords):
                return category
        return "Other"

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mteb_results_{self.config.model_name}_{timestamp}.json"

        output_path = Path(self.config.output_dir) / filename

        output = {
            "model": self.config.model_name,
            "model_path": self.config.model_path,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "results": self.results,
            "summary": self.summarize_results() if self.results else {},
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
        return output_path


def run_mteb_evaluation(
    model_path: str,
    output_dir: str = "./mteb_results",
    task_categories: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Convenience function to run MTEB evaluation.

    Args:
        model_path: Path to the extracted encoder model
        output_dir: Directory to save results
        task_categories: List of task categories to evaluate
        task_names: Specific task names (overrides categories)
        batch_size: Batch size for encoding

    Returns:
        Evaluation results dictionary
    """
    config = MTEBConfig(
        model_path=model_path,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    if task_categories:
        config.task_categories = task_categories
    if task_names:
        config.task_names = task_names

    evaluator = MTEBEvaluator(config)
    results = evaluator.run_evaluation()
    evaluator.save_results()

    return results
