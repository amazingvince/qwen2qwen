#!/usr/bin/env python3
"""Run STS-B baseline evaluation on embedding models.

Establishes reference values for comparing Qwen3-Encoder training progress.
Evaluates on STS-B and SICK test sets (no training on STS-B train data).

Usage:
    python scripts/run_baseline_stsb.py --output_dir ./baseline_results
    python scripts/run_baseline_stsb.py --output_dir ./baseline_results --plot
    python scripts/run_baseline_stsb.py --batch_size 16  # For memory-constrained runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.baseline_comparison import DecoderPoolWrapper
from evaluation.similarity_eval import STSEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineModelConfig:
    """Configuration for a baseline model."""

    name: str  # Display name
    model_id: str  # HuggingFace model ID
    model_type: str  # "sentence_transformer" | "encoder_pool"
    pooling: str = "mean"
    params: str = ""  # Parameter count for display


# Baseline models to evaluate
BASELINE_MODELS = [
    BaselineModelConfig(
        name="Qwen3-Embedding-0.6B",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_type="sentence_transformer",
        params="600M",
    ),
    BaselineModelConfig(
        name="Ettin-Encoder-1B",
        model_id="jhu-clsp/ettin-encoder-1b",
        model_type="encoder_pool",
        pooling="mean",
        params="1B",
    ),
    BaselineModelConfig(
        name="Ettin-Encoder-400M",
        model_id="jhu-clsp/ettin-encoder-400m",
        model_type="encoder_pool",
        pooling="mean",
        params="400M",
    ),
    BaselineModelConfig(
        name="RoBERTa-base",
        model_id="FacebookAI/roberta-base",
        model_type="encoder_pool",
        pooling="mean",
        params="125M",
    ),
    BaselineModelConfig(
        name="E5-base-v2",
        model_id="intfloat/e5-base-v2",
        model_type="sentence_transformer",
        params="109M",
    ),
    BaselineModelConfig(
        name="all-MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        model_type="sentence_transformer",
        params="22M",
    ),
]


def load_baseline_model(config: BaselineModelConfig, device: str = "cuda") -> Any:
    """Load a baseline model with appropriate wrapper."""
    logger.info(f"Loading {config.name} ({config.model_id})...")

    if config.model_type == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.model_id, device=device)
            logger.info("  Loaded as SentenceTransformer")
            return model
        except Exception as e:
            logger.warning(
                f"  SentenceTransformer failed ({e}), trying DecoderPoolWrapper"
            )
            return DecoderPoolWrapper(config.model_id, pooling="mean", device=device)
    else:
        # encoder_pool type
        model = DecoderPoolWrapper(
            config.model_id, pooling=config.pooling, device=device
        )
        logger.info(f"  Loaded with {config.pooling} pooling")
        return model


def evaluate_baseline(
    model: Any,
    name: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run STS-B evaluation on a single model."""
    logger.info(f"\nEvaluating {name}...")
    logger.info("=" * 50)

    evaluator = STSEvaluator(model, device=device)

    # Just evaluate STS-B (SICK has deprecated dataset script issues)
    result = evaluator.evaluate_dataset("stsb", batch_size=batch_size)

    output = {
        "stsb": {
            "spearman": result.spearman,
            "pearson": result.pearson,
            "num_samples": result.num_samples,
        }
    }
    logger.info(
        f"  STS-B: spearman={result.spearman:.4f}, pearson={result.pearson:.4f}"
    )

    return output


def run_all_baselines(
    output_dir: str,
    batch_size: int = 32,
    device: str = "cuda",
    plot: bool = False,
) -> dict[str, Any]:
    """Run STS evaluation on all baseline models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "batch_size": batch_size,
        "models": {},
    }

    for config in BASELINE_MODELS:
        try:
            model = load_baseline_model(config, device=device)
            model_results = evaluate_baseline(
                model, config.name, device=device, batch_size=batch_size
            )
            results["models"][config.name] = {
                "model_id": config.model_id,
                "params": config.params,
                "results": model_results,
            }
        except Exception as e:
            logger.error(f"Failed to evaluate {config.name}: {e}")
            results["models"][config.name] = {
                "model_id": config.model_id,
                "params": config.params,
                "error": str(e),
            }

        # Clear GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save JSON results
    json_path = output_path / "stsb_baselines.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nJSON results saved to {json_path}")

    # Generate markdown report
    md_path = output_path / "stsb_baselines.md"
    generate_markdown_report(results, md_path)
    logger.info(f"Markdown report saved to {md_path}")

    # Optional: generate chart
    if plot:
        chart_path = output_path / "stsb_chart.png"
        generate_chart(results, chart_path)

    return results


def generate_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate markdown report with results tables."""
    lines = [
        "# STS-B Baseline Evaluation Results",
        "",
        f"Generated: {results['timestamp'][:10]}",
        "",
        "## Summary",
        "",
        "These baselines establish reference values for monitoring Qwen3-Encoder training progress.",
        "Models are evaluated on STS-B test set without fine-tuning on STS-B training data.",
        "",
        "## Results",
        "",
        "| Model | Params | STS-B Spearman | STS-B Pearson |",
        "|-------|--------|----------------|---------------|",
    ]

    # Collect results for sorting
    model_rows = []
    for model_name, model_data in results["models"].items():
        if "error" in model_data:
            model_rows.append((model_name, model_data["params"], None, None))
            continue

        res = model_data["results"]
        stsb_spearman = res.get("stsb", {}).get("spearman", 0)
        stsb_pearson = res.get("stsb", {}).get("pearson", 0)

        model_rows.append(
            (model_name, model_data["params"], stsb_spearman, stsb_pearson)
        )

    # Sort by spearman (descending)
    model_rows.sort(key=lambda x: x[2] if x[2] is not None else -1, reverse=True)

    for row in model_rows:
        name, params, stsb_s, stsb_p = row
        if stsb_s is None:
            lines.append(f"| {name} | {params} | ERROR | - |")
        else:
            lines.append(f"| {name} | {params} | {stsb_s:.4f} | {stsb_p:.4f} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **Qwen3-Embedding-0.6B** is the primary target for our trained encoder to match/beat",
            "- Ettin models are encoder-only transformers (no contrastive training), evaluated with mean pooling",
            "- RoBERTa-base is a raw encoder baseline with mean pooling (no embedding training)",
            "- E5-base-v2 and all-MiniLM-L6-v2 are sentence-transformers models for reference",
            "- Spearman correlation (rho) is the primary metric for STS tasks",
            "",
            "## Evaluation Details",
            "",
            f"- **Batch size**: {results['batch_size']}",
            f"- **Device**: {results['device']}",
            "- **Dataset**: STS-B test (1379 sentence pairs)",
            "- **Metric**: Cosine similarity between sentence embeddings",
            "",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_chart(results: dict[str, Any], output_path: Path) -> None:
    """Generate bar chart of results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping chart generation")
        return

    # Extract data
    models = []
    stsb_spearman = []
    stsb_pearson = []

    for model_name, model_data in results["models"].items():
        if "error" in model_data:
            continue
        res = model_data["results"]
        models.append(model_name)
        stsb_spearman.append(res.get("stsb", {}).get("spearman", 0))
        stsb_pearson.append(res.get("stsb", {}).get("pearson", 0))

    # Sort by STS-B spearman score
    sorted_indices = np.argsort(stsb_spearman)[::-1]
    models = [models[i] for i in sorted_indices]
    stsb_spearman = [stsb_spearman[i] for i in sorted_indices]
    stsb_pearson = [stsb_pearson[i] for i in sorted_indices]

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, stsb_spearman, width, label="Spearman", color="#2196F3"
    )
    bars2 = ax.bar(x + width / 2, stsb_pearson, width, label="Pearson", color="#4CAF50")

    ax.set_ylabel("Correlation")
    ax.set_title("STS-B Baseline Evaluation Results")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run STS-B baseline evaluation on embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./docs/baselines",
        help="Directory to save results (default: ./docs/baselines)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (default: cuda)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate bar chart (requires matplotlib)",
    )

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        logger.warning("CUDA not available, using CPU")

    logger.info("=" * 60)
    logger.info("STS-B Baseline Evaluation")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Models to evaluate: {len(BASELINE_MODELS)}")
    logger.info("")

    results = run_all_baselines(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        plot=args.plot,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)

    # Print summary
    successful = sum(1 for m in results["models"].values() if "error" not in m)
    logger.info(f"Successfully evaluated: {successful}/{len(BASELINE_MODELS)} models")
    logger.info(f"Results saved to: {args.output_dir}/")

    return results


if __name__ == "__main__":
    main()
