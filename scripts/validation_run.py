#!/usr/bin/env python3
"""
Validation training run - 1B tokens to verify convergence.

Usage:
    accelerate launch scripts/validation_run.py \
        --model-path ./initialized-model \
        --output-dir ./validation-output

    # With custom steps
    accelerate launch scripts/validation_run.py \
        --model-path ./initialized-model \
        --output-dir ./validation-output \
        --num-steps 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm.auto import tqdm

from data import UL2DataCollator, t5gemma2_config
from qwen3_encdec import Qwen3EncoderDecoderTokenizer, Qwen3ForSeq2SeqLM
from training.execution import PHASE_CONFIGS, TrainingPhase
from training.monitor import TrainingMonitor, compute_gradient_norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dummy_dataset(tokenizer, size: int = 10000, seq_length: int = 256):
    """Create a dummy dataset for validation."""
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self):
            self.size = size
            self.vocab_size = tokenizer.vocab_size
            self.seq_length = seq_length

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Generate random tokens (avoiding sentinel tokens)
            input_ids = torch.randint(
                100,
                self.vocab_size - 200,
                (self.seq_length,),
            )
            return {"input_ids": input_ids}

    return DummyDataset()


def create_streaming_dataset(dataset_name: str, tokenizer, max_seq_length: int = 2048):
    """Create a streaming dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        raise

    dataset = load_dataset(
        dataset_name,
        name="sample-10BT",  # Use smaller sample for validation
        streaming=True,
        split="train",
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            return_attention_mask=False,
        )

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    return dataset


def run_validation(
    model_path: str,
    output_dir: str,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    num_steps: int = 5000,
    use_dummy_data: bool = False,
) -> bool:
    """
    Run validation training.

    Args:
        model_path: Path to initialized model.
        output_dir: Output directory for results.
        dataset_name: HuggingFace dataset name.
        num_steps: Number of training steps.
        use_dummy_data: Use dummy data instead of real dataset.

    Returns:
        True if validation passed, False otherwise.
    """
    phase_config = PHASE_CONFIGS[TrainingPhase.VALIDATION]

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
    )

    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}...")
    model = Qwen3ForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(model_path)

    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Create dataset
    logger.info("Creating dataset...")
    if use_dummy_data:
        dataset = create_dummy_dataset(tokenizer)
    else:
        dataset = create_streaming_dataset(
            dataset_name,
            tokenizer,
            max_seq_length=2048,
        )

    ul25_config = t5gemma2_config()
    collator = UL2DataCollator(
        tokenizer,
        config=ul25_config,
        max_length=1024,
        max_labels_length=512,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        num_workers=2 if not use_dummy_data else 0,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01,
    )

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training monitor
    monitor = TrainingMonitor()

    # Training loop
    logger.info(f"Starting validation run ({num_steps} steps)...")
    model.train()

    progress = tqdm(
        range(num_steps),
        disable=not accelerator.is_local_main_process,
        desc="Validation",
    )
    data_iter = iter(dataloader)

    warmup_steps = 100
    total_tokens = 0

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            # Compute metrics
            unwrapped_model = accelerator.unwrap_model(model)
            grad_norm = compute_gradient_norm(unwrapped_model)

            batch_tokens = batch["input_ids"].numel()
            if "decoder_input_ids" in batch:
                batch_tokens += batch["decoder_input_ids"].numel()
            total_tokens += batch_tokens * accelerator.num_processes

            # Update monitor
            monitor.update(
                step=step,
                loss=loss.item(),
                grad_norm=grad_norm,
                learning_rate=optimizer.param_groups[0]["lr"],
                tokens_per_second=batch_tokens / 0.5,  # Rough estimate
                is_warmup=(step < warmup_steps),
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm:.2f}",
            )

    # Validate results
    validation_results = monitor.validate_phase(phase_config)

    # Print summary
    if accelerator.is_main_process:
        summary = monitor.get_summary()

        print("\n" + "=" * 60)
        print("VALIDATION RUN SUMMARY")
        print("=" * 60)
        print(f"Steps completed: {summary['steps']}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Initial loss: {summary['loss']['initial']:.4f}")
        print(f"Final loss: {summary['loss']['current']:.4f}")
        print(f"Loss decrease: {summary['loss']['decrease']:.4f}")
        print(f"Max grad norm: {summary['grad_norm']['max']:.4f}")
        print(f"Alerts: {summary['alerts']}")
        print("=" * 60)

        print("\nValidation Checks:")
        for check in validation_results["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['name']}: {check['message']}")

        if validation_results["passed"]:
            print("\nVALIDATION PASSED - Ready for full training")
        else:
            print("\nVALIDATION FAILED - Review issues before proceeding")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "validation_results.json", "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "validation": validation_results,
                },
                f,
                indent=2,
            )

        # Save model if passed
        if validation_results["passed"]:
            model_output = output_path / "model"
            accelerator.unwrap_model(model).save_pretrained(model_output)
            tokenizer.save_pretrained(model_output)
            logger.info(f"Saved model to {model_output}")

    return validation_results["passed"]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run validation training on the model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to initialized model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help="Use dummy data instead of real dataset",
    )
    args = parser.parse_args()

    success = run_validation(
        args.model_path,
        args.output_dir,
        args.dataset,
        args.num_steps,
        args.dummy_data,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
