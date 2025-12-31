#!/usr/bin/env python3
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
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qwen3_encdec import Qwen3EncoderDecoderTokenizer
from qwen3_encdec.weight_initialization import (
    initialize_from_qwen3,
    verify_gradient_flow,
    verify_weight_initialization,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
            status = "PASS" if passed else "FAIL"
            logger.info(f"  [{status}] {check}")

        if not all(weight_results.values()):
            logger.error("Weight verification failed!")
            return 1

        logger.info("Running gradient flow verification...")
        grad_results = verify_gradient_flow(model)
        for check, passed in grad_results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  [{status}] {check}")

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
        logger.info(f"Converting model to {args.dtype}...")
        model = model.to(dtype_map[args.dtype])

    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)

    # Create and save tokenizer
    logger.info("Creating tokenizer with sentinel tokens...")
    tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(
        args.qwen3_model,
        num_sentinel_tokens=args.num_sentinel_tokens,
    )
    tokenizer.save_pretrained(output_path)
    logger.info(f"Tokenizer saved with vocab size: {tokenizer.vocab_size}")

    # Update config with special token IDs for generation
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = (
        tokenizer.pad_token_id
    )  # Use pad as decoder start
    model.config.save_pretrained(output_path)

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
    logger.info(f"Model saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
