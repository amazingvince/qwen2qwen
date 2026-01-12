"""Compare Qwen3 Encoder against baseline models.

Baselines:
- Qwen3-0.6B with mean pooling (no bidirectional training)
- Other embedding models (E5, GTE, etc.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .retrieval_eval import RetrievalEvaluator, RetrievalResult
from .similarity_eval import STSEvaluator, STSResult


@dataclass
class ModelInfo:
    """Information about a model for comparison."""

    name: str
    path: str
    model_type: str  # "qwen3_encoder", "decoder_pool", "sentence_transformer"
    pooling: str = "mean"
    normalize: bool = True


class DecoderPoolWrapper:
    """
    Wrapper for decoder-only models with mean pooling.
    Used as baseline comparison (Qwen3-0.6B without bidirectional training).
    """

    def __init__(
        self,
        model_path: str,
        pooling: str = "mean",
        device: str = "cuda",
        max_seq_length: int = 512,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pooling = pooling
        self.max_seq_length = max_seq_length

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences using decoder with pooling."""
        all_embeddings = []

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for start_idx in iterator:
                batch = sentences[start_idx : start_idx + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state

                # Apply pooling
                if self.pooling == "mean":
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    embeddings = (hidden_states * attention_mask).sum(
                        1
                    ) / attention_mask.sum(1)
                elif self.pooling == "last":
                    # Get last non-padded token
                    seq_lengths = inputs["attention_mask"].sum(1) - 1
                    embeddings = hidden_states[
                        torch.arange(hidden_states.size(0), device=self.device),
                        seq_lengths,
                    ]
                elif self.pooling == "cls":
                    embeddings = hidden_states[:, 0]
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings


class BaselineComparison:
    """Compare multiple embedding models on the same benchmarks."""

    # Pre-defined baseline models
    BASELINES = {
        "e5-base": ModelInfo(
            name="E5-base",
            path="intfloat/e5-base-v2",
            model_type="sentence_transformer",
        ),
        "gte-base": ModelInfo(
            name="GTE-base",
            path="thenlper/gte-base",
            model_type="sentence_transformer",
        ),
        "bge-base": ModelInfo(
            name="BGE-base",
            path="BAAI/bge-base-en-v1.5",
            model_type="sentence_transformer",
        ),
    }

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def load_model(self, model_info: ModelInfo) -> Any:
        """Load a model for evaluation."""
        if model_info.model_type == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_info.path, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for baseline models. "
                    "Install with: pip install sentence-transformers"
                )
        elif model_info.model_type == "decoder_pool":
            model = DecoderPoolWrapper(
                model_info.path,
                pooling=model_info.pooling,
                device=self.device,
            )
        elif model_info.model_type == "qwen3_encoder":
            from .mteb_eval import Qwen3EncoderWrapper

            model = Qwen3EncoderWrapper(
                model_info.path,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown model type: {model_info.model_type}")

        self.models[model_info.name] = model
        return model

    def add_model(
        self,
        name: str,
        path: str,
        model_type: str = "sentence_transformer",
        **kwargs,
    ) -> None:
        """Add a model for comparison."""
        model_info = ModelInfo(name=name, path=path, model_type=model_type, **kwargs)
        self.load_model(model_info)

    def add_qwen3_baseline(self, qwen3_path: str = "Qwen/Qwen3-0.6B") -> None:
        """Add Qwen3-0.6B with mean pooling as baseline."""
        model_info = ModelInfo(
            name="Qwen3-0.6B (mean pool)",
            path=qwen3_path,
            model_type="decoder_pool",
            pooling="mean",
        )
        self.load_model(model_info)

    def add_trained_encoder(
        self, encoder_path: str, name: str = "Qwen3-Encoder (ours)"
    ) -> None:
        """Add our trained encoder."""
        model_info = ModelInfo(
            name=name,
            path=encoder_path,
            model_type="qwen3_encoder",
        )
        self.load_model(model_info)

    def run_sts_comparison(
        self, batch_size: int = 32
    ) -> Dict[str, Dict[str, STSResult]]:
        """Run STS evaluation on all models."""
        results = {}

        for name, model in self.models.items():
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {name}")
            print("=" * 60)

            evaluator = STSEvaluator(model, device=self.device)
            results[name] = evaluator.evaluate_all(batch_size)

        self.results["sts"] = results
        return results

    def run_retrieval_comparison(
        self,
        queries: List[str],
        documents: List[str],
        relevant_docs: List[List[int]],
        batch_size: int = 32,
    ) -> Dict[str, RetrievalResult]:
        """Run retrieval evaluation on all models."""
        results = {}

        for name, model in self.models.items():
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {name}")
            print("=" * 60)

            evaluator = RetrievalEvaluator(model, device=self.device)
            results[name] = evaluator.evaluate_custom(
                queries, documents, relevant_docs, batch_size
            )

        self.results["retrieval"] = results
        return results

    def create_comparison_table(self) -> str:
        """Create a formatted comparison table."""
        lines = []

        # STS Results
        if "sts" in self.results:
            lines.append("\n## Semantic Textual Similarity (Spearman Correlation)")
            lines.append("")

            # Header
            sts_results = self.results["sts"]
            datasets = list(next(iter(sts_results.values())).keys())
            header = "| Model | " + " | ".join(datasets) + " | Avg |"
            lines.append(header)
            lines.append("|" + "|".join(["---"] * (len(datasets) + 2)) + "|")

            # Rows
            for model_name, model_results in sts_results.items():
                scores = [model_results[d].spearman for d in datasets]
                avg = np.mean(scores)
                row = (
                    f"| {model_name} | "
                    + " | ".join(f"{s:.4f}" for s in scores)
                    + f" | {avg:.4f} |"
                )
                lines.append(row)

        # Retrieval Results
        if "retrieval" in self.results:
            lines.append("\n## Retrieval")
            lines.append("")

            retrieval_results = self.results["retrieval"]
            header = "| Model | MRR@10 | R@1 | R@10 | R@100 | NDCG@10 |"
            lines.append(header)
            lines.append("|---|---|---|---|---|---|")

            for model_name, result in retrieval_results.items():
                row = f"| {model_name} | {result.mrr_at_10:.4f} | {result.recall_at_1:.4f} | {result.recall_at_10:.4f} | {result.recall_at_100:.4f} | {result.ndcg_at_10:.4f} |"
                lines.append(row)

        return "\n".join(lines)

    def save_results(self, output_path: str) -> None:
        """Save all results to JSON."""
        output = {
            "models": list(self.models.keys()),
            "results": {},
        }

        if "sts" in self.results:
            output["results"]["sts"] = {
                model: {
                    dataset: asdict(result) for dataset, result in model_results.items()
                }
                for model, model_results in self.results["sts"].items()
            }

        if "retrieval" in self.results:
            output["results"]["retrieval"] = {
                model: asdict(result)
                for model, result in self.results["retrieval"].items()
            }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_path}")
