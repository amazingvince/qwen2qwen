#!/usr/bin/env python3
"""Run evaluation on extracted Qwen3 Encoder.

This script provides a CLI for running various evaluations:
- MTEB benchmark (full or specific tasks)
- STS evaluation (STS-B, SICK)
- Retrieval evaluation
- Baseline comparison

Usage:
    # Run full MTEB evaluation
    python scripts/run_evaluation.py --encoder_path ./extracted_encoder --run_mteb

    # Run STS evaluation only
    python scripts/run_evaluation.py --encoder_path ./extracted_encoder --run_sts

    # Compare against baselines
    python scripts/run_evaluation.py --encoder_path ./extracted_encoder --compare_baselines

    # Run all evaluations
    python scripts/run_evaluation.py --encoder_path ./extracted_encoder --run_all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_mteb_evaluation(
    encoder_path: str,
    output_dir: str,
    task_categories: list[str] | None = None,
    task_names: list[str] | None = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run MTEB evaluation."""
    from evaluation.mteb_eval import MTEBConfig, MTEBEvaluator

    logger.info("=" * 60)
    logger.info("Running MTEB Evaluation")
    logger.info("=" * 60)

    config = MTEBConfig(
        model_path=encoder_path,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
    )

    if task_categories:
        config.task_categories = task_categories
    if task_names:
        config.task_names = task_names

    evaluator = MTEBEvaluator(config)
    results = evaluator.run_evaluation()
    evaluator.save_results()

    summary = evaluator.summarize_results()
    logger.info("\nMTEB Summary:")
    for category, score in summary.items():
        logger.info(f"  {category}: {score:.4f}")

    return {"results": results, "summary": summary}


def run_sts_evaluation(
    encoder_path: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run STS evaluation."""
    from evaluation.mteb_eval import Qwen3EncoderWrapper
    from evaluation.similarity_eval import STSEvaluator

    logger.info("=" * 60)
    logger.info("Running STS Evaluation")
    logger.info("=" * 60)

    # Load model
    model = Qwen3EncoderWrapper(encoder_path, device=device)

    # Run evaluation
    evaluator = STSEvaluator(model, device=device)
    results = evaluator.evaluate_all(batch_size=batch_size)

    # Save results
    output_path = Path(output_dir) / "sts_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model_path": encoder_path,
        "timestamp": datetime.now().isoformat(),
        "results": {
            name: {
                "spearman": result.spearman,
                "pearson": result.pearson,
                "num_samples": result.num_samples,
            }
            for name, result in results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\nSTS Results:")
    for name, result in results.items():
        logger.info(
            f"  {name}: spearman={result.spearman:.4f}, pearson={result.pearson:.4f}"
        )

    return {"results": results}


def run_retrieval_evaluation(
    encoder_path: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run retrieval evaluation with sample data."""
    from evaluation.mteb_eval import Qwen3EncoderWrapper
    from evaluation.retrieval_eval import RetrievalEvaluator

    logger.info("=" * 60)
    logger.info("Running Retrieval Evaluation (Sample)")
    logger.info("=" * 60)

    # Load model
    model = Qwen3EncoderWrapper(encoder_path, device=device)

    # Create sample retrieval task
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is natural language processing?",
        "Explain deep learning",
        "What are transformers in AI?",
    ]

    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Natural language processing is a field of AI focused on the interaction between computers and human language.",
        "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers.",
        "Transformers are a type of neural network architecture that uses self-attention mechanisms.",
        "Python is a popular programming language for data science.",
        "The weather today is sunny with clear skies.",
        "Cats are popular pets known for their independence.",
        "Supervised learning uses labeled data for training models.",
        "Reinforcement learning involves learning through trial and error with rewards.",
    ]

    # Each query's relevant document indices
    relevant_docs = [
        [0],  # "What is machine learning?" -> doc 0
        [1],  # "How do neural networks work?" -> doc 1
        [2],  # "What is natural language processing?" -> doc 2
        [3],  # "Explain deep learning" -> doc 3
        [4],  # "What are transformers in AI?" -> doc 4
    ]

    evaluator = RetrievalEvaluator(model, device=device)
    result = evaluator.evaluate_custom(
        queries=queries,
        documents=documents,
        relevant_docs=relevant_docs,
        batch_size=batch_size,
    )

    # Save results
    output_path = Path(output_dir) / "retrieval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model_path": encoder_path,
        "timestamp": datetime.now().isoformat(),
        "results": {
            "mrr_at_10": result.mrr_at_10,
            "recall_at_1": result.recall_at_1,
            "recall_at_10": result.recall_at_10,
            "recall_at_100": result.recall_at_100,
            "ndcg_at_10": result.ndcg_at_10,
            "num_queries": result.num_queries,
            "num_docs": result.num_docs,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"\n{result}")

    return {"result": result}


def run_baseline_comparison(
    encoder_path: str,
    output_dir: str,
    qwen3_baseline_path: str | None = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run baseline comparison."""
    from evaluation.baseline_comparison import BaselineComparison

    logger.info("=" * 60)
    logger.info("Running Baseline Comparison")
    logger.info("=" * 60)

    comparison = BaselineComparison(device=device)

    # Add our trained encoder
    comparison.add_trained_encoder(encoder_path, name="Qwen3-Encoder (ours)")

    # Add Qwen3 baseline (decoder with mean pooling)
    if qwen3_baseline_path:
        comparison.add_qwen3_baseline(qwen3_baseline_path)
    else:
        logger.info("Skipping Qwen3 baseline (no path provided)")

    # Run STS comparison
    try:
        sts_results = comparison.run_sts_comparison(batch_size=batch_size)
    except Exception as e:
        logger.warning(f"STS comparison failed: {e}")
        sts_results = None

    # Create comparison table
    table = comparison.create_comparison_table()
    logger.info("\n" + table)

    # Save results
    output_path = Path(output_dir) / "baseline_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save_results(str(output_path))

    # Save table as markdown
    table_path = Path(output_dir) / "baseline_comparison.md"
    with open(table_path, "w") as f:
        f.write("# Baseline Comparison Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(table)

    logger.info(f"\nTable saved to {table_path}")

    return {"sts_results": sts_results, "table": table}


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on extracted Qwen3 Encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to the extracted encoder model",
    )

    # Evaluation types
    parser.add_argument(
        "--run_mteb",
        action="store_true",
        help="Run full MTEB evaluation",
    )
    parser.add_argument(
        "--run_sts",
        action="store_true",
        help="Run STS evaluation (STS-B, SICK)",
    )
    parser.add_argument(
        "--run_retrieval",
        action="store_true",
        help="Run retrieval evaluation",
    )
    parser.add_argument(
        "--compare_baselines",
        action="store_true",
        help="Compare against baseline models",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all evaluations",
    )

    # MTEB specific options
    parser.add_argument(
        "--mteb_tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific MTEB tasks to run",
    )
    parser.add_argument(
        "--mteb_categories",
        type=str,
        nargs="+",
        default=None,
        help="MTEB task categories to run",
    )

    # Baseline comparison options
    parser.add_argument(
        "--qwen3_baseline",
        type=str,
        default=None,
        help="Path to Qwen3 model for baseline comparison",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save results",
    )

    # Hardware options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda/cpu)",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Check if any evaluation is requested
    if not any(
        [
            args.run_mteb,
            args.run_sts,
            args.run_retrieval,
            args.compare_baselines,
            args.run_all,
        ]
    ):
        parser.error("At least one evaluation type must be specified")

    results = {}

    # Run evaluations
    if args.run_all or args.run_mteb:
        try:
            results["mteb"] = run_mteb_evaluation(
                encoder_path=args.encoder_path,
                output_dir=args.output_dir,
                task_categories=args.mteb_categories,
                task_names=args.mteb_tasks,
                batch_size=args.batch_size,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"MTEB evaluation failed: {e}")
            results["mteb"] = {"error": str(e)}

    if args.run_all or args.run_sts:
        try:
            results["sts"] = run_sts_evaluation(
                encoder_path=args.encoder_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"STS evaluation failed: {e}")
            results["sts"] = {"error": str(e)}

    if args.run_all or args.run_retrieval:
        try:
            results["retrieval"] = run_retrieval_evaluation(
                encoder_path=args.encoder_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {e}")
            results["retrieval"] = {"error": str(e)}

    if args.run_all or args.compare_baselines:
        try:
            results["baseline"] = run_baseline_comparison(
                encoder_path=args.encoder_path,
                output_dir=args.output_dir,
                qwen3_baseline_path=args.qwen3_baseline,
                batch_size=args.batch_size,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"Baseline comparison failed: {e}")
            results["baseline"] = {"error": str(e)}

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output_dir}")

    return results


if __name__ == "__main__":
    main()
