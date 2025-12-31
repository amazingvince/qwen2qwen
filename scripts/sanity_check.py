#!/usr/bin/env python3
"""
Sanity check script - verify model forward/backward works.

Usage:
    python scripts/sanity_check.py --model-path ./initialized-model

    # CPU only (for testing without GPU)
    python scripts/sanity_check.py --model-path ./initialized-model --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from qwen3_encdec import Qwen3EncoderDecoderTokenizer, Qwen3ForSeq2SeqLM
from training.memory_utils import get_memory_stats
from training.monitor import compute_gradient_norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dummy_batch(
    tokenizer: Any,
    batch_size: int = 2,
    enc_len: int = 128,
    dec_len: int = 64,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Create a dummy batch for testing."""
    vocab_size = tokenizer.vocab_size

    encoder_input_ids = torch.randint(100, vocab_size - 200, (batch_size, enc_len))
    decoder_input_ids = torch.randint(100, vocab_size - 200, (batch_size, dec_len))
    labels = torch.randint(100, vocab_size - 200, (batch_size, dec_len))

    # Set some labels to -100 (ignored)
    labels[:, -10:] = -100

    batch = {
        "input_ids": encoder_input_ids.to(device),
        "attention_mask": torch.ones_like(encoder_input_ids).to(device),
        "decoder_input_ids": decoder_input_ids.to(device),
        "decoder_attention_mask": torch.ones_like(decoder_input_ids).to(device),
        "labels": labels.to(device),
    }

    return batch


def run_sanity_check(model_path: str, device: str = "cuda") -> Dict[str, Any]:
    """
    Run comprehensive sanity checks on the model.

    Checks:
    1. Model loads correctly
    2. Forward pass works
    3. Loss is computed
    4. Backward pass works
    5. Gradients are non-zero
    6. Tied embeddings work
    7. Generation works

    Args:
        model_path: Path to saved model.
        device: Device to run on ("cuda" or "cpu").

    Returns:
        Dictionary with check results.
    """
    results: Dict[str, Any] = {"passed": True, "checks": []}

    # 1. Load model
    logger.info("Loading model...")
    try:
        model = Qwen3ForSeq2SeqLM.from_pretrained(model_path)
        model = model.to(device)
        results["checks"].append({"name": "model_load", "passed": True})
    except Exception as e:
        results["checks"].append(
            {"name": "model_load", "passed": False, "error": str(e)}
        )
        results["passed"] = False
        return results

    # Load tokenizer
    try:
        tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(model_path)
        results["checks"].append({"name": "tokenizer_load", "passed": True})
    except Exception as e:
        results["checks"].append(
            {"name": "tokenizer_load", "passed": False, "error": str(e)}
        )
        results["passed"] = False
        return results

    # 2. Forward pass
    logger.info("Testing forward pass...")
    batch = create_dummy_batch(tokenizer, device=device)

    try:
        model.train()
        outputs = model(**batch)
        results["checks"].append(
            {
                "name": "forward_pass",
                "passed": True,
                "loss": outputs.loss.item(),
            }
        )
    except Exception as e:
        results["checks"].append(
            {"name": "forward_pass", "passed": False, "error": str(e)}
        )
        results["passed"] = False
        return results

    # 3. Loss sanity
    loss_value = outputs.loss.item()
    if not (0 < loss_value < 100):
        results["checks"].append(
            {
                "name": "loss_sanity",
                "passed": False,
                "message": f"Loss {loss_value} outside reasonable range",
            }
        )
        results["passed"] = False
    else:
        results["checks"].append(
            {
                "name": "loss_sanity",
                "passed": True,
                "loss": loss_value,
            }
        )

    # 4. Backward pass
    logger.info("Testing backward pass...")
    try:
        outputs.loss.backward()
        results["checks"].append({"name": "backward_pass", "passed": True})
    except Exception as e:
        results["checks"].append(
            {"name": "backward_pass", "passed": False, "error": str(e)}
        )
        results["passed"] = False
        return results

    # 5. Gradient checks
    logger.info("Checking gradients...")

    # Check encoder has gradients
    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.model.encoder.parameters()
    )
    results["checks"].append(
        {
            "name": "encoder_gradients",
            "passed": encoder_has_grad,
        }
    )
    if not encoder_has_grad:
        results["passed"] = False

    # Check decoder has gradients
    decoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.model.decoder.parameters()
    )
    results["checks"].append(
        {
            "name": "decoder_gradients",
            "passed": decoder_has_grad,
        }
    )
    if not decoder_has_grad:
        results["passed"] = False

    # Check gradient norm
    grad_norm = compute_gradient_norm(model)
    results["checks"].append(
        {
            "name": "gradient_norm",
            "passed": 0 < grad_norm < 1000,
            "value": grad_norm,
        }
    )

    # 6. Tied embeddings
    logger.info("Checking tied embeddings...")
    tied = (
        model.model.shared.weight.data_ptr()
        == model.model.encoder.embed_tokens.weight.data_ptr()
        == model.model.decoder.embed_tokens.weight.data_ptr()
    )
    results["checks"].append(
        {
            "name": "tied_embeddings",
            "passed": tied,
        }
    )
    if not tied:
        results["passed"] = False

    # 7. Generation
    logger.info("Testing generation...")
    model.zero_grad()
    model.eval()

    try:
        with torch.no_grad():
            encoder_inputs = batch["input_ids"][:1]  # Single example
            generated = model.generate(
                input_ids=encoder_inputs,
                max_new_tokens=20,
                num_beams=1,
            )
        results["checks"].append(
            {
                "name": "generation",
                "passed": True,
                "output_length": generated.shape[1],
            }
        )
    except Exception as e:
        results["checks"].append(
            {
                "name": "generation",
                "passed": False,
                "error": str(e),
            }
        )

    # Memory stats
    mem_stats = get_memory_stats()
    results["memory"] = mem_stats

    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run sanity checks on a trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    args = parser.parse_args()

    results = run_sanity_check(args.model_path, args.device)

    # Print results
    print("\n" + "=" * 60)
    print("SANITY CHECK RESULTS")
    print("=" * 60)

    for check in results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}")
        if "error" in check:
            print(f"         Error: {check['error']}")
        if "value" in check:
            print(f"         Value: {check['value']}")
        if "loss" in check:
            print(f"         Loss: {check['loss']:.4f}")

    print("=" * 60)
    if results["passed"]:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
