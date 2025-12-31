#!/usr/bin/env python3
"""
Main training script for Qwen3 Encoder-Decoder.

Usage:
    accelerate launch scripts/train.py --config configs/training_config.yaml

    # Resume from checkpoint
    accelerate launch scripts/train.py --config configs/training_config.yaml \\
        --resume_from ./output/checkpoint-step-10000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from data import create_collator_from_config
from qwen3_encdec import Qwen3EncoderDecoderTokenizer, Qwen3ForSeq2SeqLM
from qwen3_encdec.weight_initialization import initialize_from_qwen3
from training import FullConfig, Qwen3EncoderDecoderTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dummy_dataset(tokenizer, config):
    """Create a dummy dataset for testing."""
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, size=1000):
            self.size = size
            self.vocab_size = tokenizer.vocab_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Generate random tokens (avoiding sentinel tokens)
            seq_len = min(256, config.data.max_seq_length)
            input_ids = torch.randint(
                100,
                self.vocab_size - 200,
                (seq_len,),
            )
            return {"input_ids": input_ids}

    return DummyDataset()


def create_streaming_dataset(dataset_name, tokenizer, config):
    """Create a streaming dataset from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        raise

    dataset = load_dataset(
        dataset_name,
        config.data.dataset_config,
        streaming=config.data.streaming,
        split="train",
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples[config.data.text_column],
            truncation=True,
            max_length=config.data.max_seq_length,
            return_attention_mask=False,
        )

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    if config.data.streaming:
        dataset = dataset.shuffle(
            seed=config.training.seed,
            buffer_size=config.data.shuffle_buffer_size,
        )

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3 Encoder-Decoder model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--dummy_data",
        action="store_true",
        help="Use dummy data for testing",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = FullConfig.from_yaml(args.config)

    if args.output_dir:
        config.infra.output_dir = args.output_dir

    # Create output directory
    output_path = Path(config.infra.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config copy
    config.to_yaml(str(output_path / "config.yaml"))

    # Load or initialize model
    model_path = Path(config.model.model_name_or_path)
    config_file = model_path / "config.json" if model_path.exists() else None

    # Check if this is already our encoder-decoder format
    is_encdec = False
    if config_file and config_file.exists():
        import json
        with open(config_file) as f:
            model_config = json.load(f)
        is_encdec = model_config.get("model_type") == "qwen3_encdec"

    if is_encdec:
        logger.info(f"Loading pre-initialized model from {config.model.model_name_or_path}...")
        model = Qwen3ForSeq2SeqLM.from_pretrained(config.model.model_name_or_path)
    else:
        logger.info(f"Initializing model from Qwen3 checkpoint {config.model.model_name_or_path}...")
        model = initialize_from_qwen3(
            config.model.model_name_or_path,
            num_sentinel_tokens=config.model.num_sentinel_tokens,
        )

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = Qwen3EncoderDecoderTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        num_sentinel_tokens=config.model.num_sentinel_tokens,
    )

    # Create dataset
    logger.info("Creating dataset...")
    if args.dummy_data:
        train_dataset = create_dummy_dataset(tokenizer, config)
    else:
        train_dataset = create_streaming_dataset(
            config.data.dataset_name,
            tokenizer,
            config,
        )

    # Create UL2 data collator (uses UL25Config.recommended() by default)
    collator = create_collator_from_config(tokenizer, config.data)

    # Create data loader
    num_workers = config.data.preprocessing_num_workers
    dataloader_kwargs = {
        "batch_size": config.training.per_device_train_batch_size,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": config.data.dataloader_pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.data.dataloader_prefetch_factor
        dataloader_kwargs["persistent_workers"] = (
            config.data.dataloader_persistent_workers
        )
    train_dataloader = DataLoader(
        train_dataset,
        **dataloader_kwargs,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Qwen3EncoderDecoderTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataloader=train_dataloader,
        collator=collator,  # Enables curriculum progress updates
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Training complete! Model saved to {config.infra.output_dir}")


if __name__ == "__main__":
    main()
