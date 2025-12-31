#!/usr/bin/env python3
"""Quick sanity check for extracted Qwen3 Encoder.

Runs fast tests to verify the encoder is working correctly:
1. Basic encoding test
2. Semantic similarity check
3. Embedding diversity check
4. Optional: Quick STS-B evaluation

Usage:
    python scripts/quick_eval.py --encoder_path ./extracted_encoder
    python scripts/quick_eval.py --encoder_path ./extracted_encoder --run_stsb
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def check_basic_encoding(model, verbose: bool = True) -> bool:
    """Test basic encoding functionality."""
    if verbose:
        print("\n" + "=" * 50)
        print("Test 1: Basic Encoding")
        print("=" * 50)

    sentences = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    try:
        embeddings = model.encode(sentences, show_progress_bar=False)

        if verbose:
            print(f"  Input: {len(sentences)} sentences")
            print(f"  Output shape: {embeddings.shape}")
            print(f"  Embedding dim: {embeddings.shape[1]}")
            print(f"  Dtype: {embeddings.dtype}")

        # Check shape
        assert embeddings.shape[0] == len(sentences), "Wrong number of embeddings"
        assert embeddings.shape[1] > 0, "Zero-dimensional embeddings"
        assert not np.isnan(embeddings).any(), "NaN values in embeddings"
        assert not np.isinf(embeddings).any(), "Inf values in embeddings"

        if verbose:
            print("  ✓ PASSED")
        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ FAILED: {e}")
        return False


def check_semantic_similarity(model, verbose: bool = True) -> bool:
    """Test semantic similarity - similar sentences should have higher similarity."""
    if verbose:
        print("\n" + "=" * 50)
        print("Test 2: Semantic Similarity")
        print("=" * 50)

    # Similar pairs should have higher similarity than dissimilar pairs
    test_cases = [
        {
            "anchor": "The cat sat on the mat.",
            "similar": "A cat is sitting on a mat.",
            "dissimilar": "Python is a programming language.",
        },
        {
            "anchor": "Machine learning models learn from data.",
            "similar": "ML algorithms train on datasets.",
            "dissimilar": "The weather is nice today.",
        },
        {
            "anchor": "Neural networks have multiple layers.",
            "similar": "Deep learning uses layered neural architectures.",
            "dissimilar": "I like pizza for dinner.",
        },
    ]

    try:
        all_passed = True

        for i, case in enumerate(test_cases):
            sentences = [case["anchor"], case["similar"], case["dissimilar"]]
            embeddings = model.encode(sentences, show_progress_bar=False)

            # Normalize for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Compute similarities
            sim_similar = np.dot(embeddings[0], embeddings[1])
            sim_dissimilar = np.dot(embeddings[0], embeddings[2])

            passed = sim_similar > sim_dissimilar

            if verbose:
                status = "✓" if passed else "✗"
                print(f"\n  Case {i + 1}:")
                print(f'    Anchor: "{case["anchor"][:40]}..."')
                print(f"    Similar: {sim_similar:.4f}")
                print(f"    Dissimilar: {sim_dissimilar:.4f}")
                print(f"    {status} Similar > Dissimilar: {passed}")

            if not passed:
                all_passed = False

        if verbose:
            print(f"\n  {'✓ PASSED' if all_passed else '✗ FAILED'}")

        return all_passed

    except Exception as e:
        if verbose:
            print(f"  ✗ FAILED: {e}")
        return False


def check_embedding_diversity(model, verbose: bool = True) -> bool:
    """Check that different sentences produce different embeddings."""
    if verbose:
        print("\n" + "=" * 50)
        print("Test 3: Embedding Diversity")
        print("=" * 50)

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",  # Paraphrase
        "Python is a popular programming language.",
        "The stock market closed higher today.",
        "Neural networks can learn complex patterns.",
        "I love eating pizza on Friday nights.",
        "The sun rises in the east.",
        "Quantum computing uses qubits for computation.",
    ]

    try:
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute pairwise similarities
        similarities = embeddings @ embeddings.T

        # Get off-diagonal similarities
        n = len(sentences)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag_sims = similarities[off_diag_mask]

        mean_sim = np.mean(off_diag_sims)
        max_sim = np.max(off_diag_sims)
        min_sim = np.min(off_diag_sims)

        if verbose:
            print(f"  Sentences: {n}")
            print(f"  Mean similarity: {mean_sim:.4f}")
            print(f"  Max similarity: {max_sim:.4f}")
            print(f"  Min similarity: {min_sim:.4f}")

        # Check that embeddings are diverse (not all identical)
        # Paraphrases should be similar, but unrelated sentences should differ
        assert max_sim < 0.99, "Embeddings too similar (possible collapse)"
        assert mean_sim < 0.9, "Average similarity too high"
        assert min_sim < max_sim, "No variation in similarities"

        if verbose:
            print("  ✓ PASSED")
        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ FAILED: {e}")
        return False


def check_normalization(model, verbose: bool = True) -> bool:
    """Check that normalized embeddings have unit norm."""
    if verbose:
        print("\n" + "=" * 50)
        print("Test 4: Normalization")
        print("=" * 50)

    sentences = [
        "This is a test sentence.",
        "Another example for testing.",
        "Machine learning is fascinating.",
    ]

    try:
        # Get normalized embeddings
        embeddings = model.encode(
            sentences,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        norms = np.linalg.norm(embeddings, axis=1)

        if verbose:
            print(f"  Norms: {norms}")
            print(f"  Mean norm: {np.mean(norms):.6f}")
            print(f"  Std norm: {np.std(norms):.6f}")

        # Check all norms are approximately 1
        assert np.allclose(norms, 1.0, atol=1e-5), f"Norms not unit: {norms}"

        if verbose:
            print("  ✓ PASSED")
        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ FAILED: {e}")
        return False


def run_quick_stsb(model, verbose: bool = True) -> float | None:
    """Run quick STS-B evaluation."""
    if verbose:
        print("\n" + "=" * 50)
        print("Test 5: STS-B Evaluation (Optional)")
        print("=" * 50)

    try:
        from evaluation.similarity_eval import STSEvaluator

        evaluator = STSEvaluator(model)
        result = evaluator.evaluate_dataset("stsb", batch_size=32)

        if verbose:
            print(f"  Spearman: {result.spearman:.4f}")
            print(f"  Pearson: {result.pearson:.4f}")
            print(f"  Samples: {result.num_samples}")
            print("  ✓ COMPLETED")

        return result.spearman

    except ImportError as e:
        if verbose:
            print(f"  ⚠ Skipped (missing dependency): {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"  ✗ FAILED: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Quick sanity check for Qwen3 Encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to the extracted encoder model",
    )
    parser.add_argument(
        "--run_stsb",
        action="store_true",
        help="Also run STS-B evaluation (slower)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 50)
    print("Quick Evaluation for Qwen3 Encoder")
    print("=" * 50)
    print(f"Encoder: {args.encoder_path}")
    print(f"Device: {args.device}")

    # Load model
    print("\nLoading model...")
    try:
        from evaluation.mteb_eval import Qwen3EncoderWrapper

        model = Qwen3EncoderWrapper(
            args.encoder_path,
            device=args.device,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Run tests
    results = {
        "basic_encoding": check_basic_encoding(model, verbose),
        "semantic_similarity": check_semantic_similarity(model, verbose),
        "embedding_diversity": check_embedding_diversity(model, verbose),
        "normalization": check_normalization(model, verbose),
    }

    if args.run_stsb:
        stsb_score = run_quick_stsb(model, verbose)
        results["stsb_spearman"] = stsb_score

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not None)

    for test, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        elif isinstance(result, float):
            status = f"Score: {result:.4f}"
        else:
            status = "⚠ SKIPPED"
        print(f"  {test}: {status}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Encoder is ready for use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the encoder.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
