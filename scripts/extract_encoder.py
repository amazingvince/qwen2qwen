#!/usr/bin/env python3
"""
Script to extract encoder from trained encoder-decoder checkpoint.

Usage:
    python scripts/extract_encoder.py \
        --checkpoint checkpoints/qwen3-encdec-final \
        --output models/qwen3-encoder \
        --pooling mean

    # With checkpoint averaging first
    python scripts/extract_encoder.py \
        --checkpoint checkpoints/ \
        --output models/qwen3-encoder \
        --average-last 5

    # Using best checkpoints by metric
    python scripts/extract_encoder.py \
        --checkpoint checkpoints/ \
        --output models/qwen3-encoder \
        --average-best 5 \
        --metric eval_loss
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract encoder from trained encoder-decoder model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained encoder-decoder checkpoint (or checkpoint directory for averaging)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save extracted encoder",
    )

    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "last", "weighted_mean"],
        help="Pooling strategy for sentence embeddings",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't L2-normalize embeddings",
    )

    # Checkpoint averaging options
    parser.add_argument(
        "--average-last",
        type=int,
        default=None,
        metavar="N",
        help="Average last N checkpoints before extraction",
    )

    parser.add_argument(
        "--average-best",
        type=int,
        default=None,
        metavar="N",
        help="Average best N checkpoints by metric before extraction",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="eval_loss",
        help="Metric for best checkpoint selection (default: eval_loss)",
    )

    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Higher metric is better (default: lower is better)",
    )

    # Sentence-transformers export
    parser.add_argument(
        "--sentence-transformers",
        action="store_true",
        help="Create sentence-transformers compatible config",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max sequence length for sentence-transformers config",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
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
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    # Handle checkpoint averaging
    if args.average_last or args.average_best:
        from extraction.checkpoint_averaging import CheckpointAverager

        averaged_path = Path(args.output).parent / "averaged_checkpoint"

        averager = CheckpointAverager(
            checkpoint_dir=str(checkpoint_path),
            output_path=str(averaged_path),
        )

        try:
            if args.average_last:
                logger.info(f"Averaging last {args.average_last} checkpoints...")
                checkpoint_path = averager.average_last_n(n=args.average_last)
            else:
                logger.info(
                    f"Averaging best {args.average_best} checkpoints by {args.metric}..."
                )
                checkpoint_path = averager.average_best_n(
                    n=args.average_best,
                    metric=args.metric,
                    lower_is_better=not args.higher_is_better,
                )
        except Exception as e:
            logger.error(f"Checkpoint averaging failed: {e}")
            return 1

    # Create extractor
    from extraction.extract_encoder import EncoderExtractor

    extractor = EncoderExtractor(
        checkpoint_path=str(checkpoint_path),
        output_path=args.output,
        pooling_mode=args.pooling,
        normalize_embeddings=not args.no_normalize,
    )

    # Extract and save
    try:
        encoder = extractor.extract_and_save()
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Create sentence-transformers config if requested
    if args.sentence_transformers:
        from extraction.sentence_transformers_export import (
            create_sentence_transformers_config,
            verify_sentence_transformers_loading,
        )

        logger.info("Creating sentence-transformers configuration...")
        create_sentence_transformers_config(
            output_path=args.output,
            hidden_size=encoder.config.hidden_size,
            max_seq_length=args.max_seq_length,
            pooling_mode=args.pooling,
            normalize=not args.no_normalize,
        )

        # Verify loading
        verify_sentence_transformers_loading(args.output)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
