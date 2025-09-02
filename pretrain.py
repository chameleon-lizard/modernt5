import os
import argparse
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    set_seed,
    TrainerCallback,
    get_scheduler,
)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from modeling_modernt5 import ModernT5ForConditionalGeneration
from collator import UL2MoDCollator
import re
from collections import defaultdict
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train ModernT5ForConditionalGeneration model")
    parser.add_argument("--model_path", type=str, default="modernt5_from_rumodernbert",
                        help="Path to the model checkpoint directory")
    parser.add_argument("--dataset_dir", type=str, default="final_pretrain_mix_tokenized",
                        help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=6,
                        help="Number of training epochs")
    parser.add_argument("--encoder_freeze_epochs", type=int, default=0,
                        help="Number of epochs to keep encoder frozen")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set wandb environment variables for Trainer integration
    os.environ["WANDB_PROJECT"] = "ModernT5"
    os.environ["WANDB_RUN_NAME"] = "modernt5_pretrain_test_lr"
    os.environ["WANDB_TAGS"] = "pretrain"
    os.environ["WANDB_WATCH"] = "gradients"

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.eos_token_id is None:
        # BERT-style models often use sep_token as the eos token.
        # The decoder config in model.py also uses sep_token_id for eos_token_id.
        print("Tokenizer does not have an EOS token. Setting eos_token to sep_token.")
        tokenizer.eos_token = tokenizer.sep_token

    # Ensure tokenizer has sentinel tokens used for UL2-style span corruption.
    required_sentinels = 100
    existing_special_tokens = tokenizer.additional_special_tokens or []
    existing_sentinels = [
        tok for tok in existing_special_tokens
        if "extra_id" in tok or "sentinel" in tok.lower()
    ]
    if len(existing_sentinels) < required_sentinels:
        sentinel_tokens = [f"<extra_id_{i}>" for i in range(required_sentinels)]
        tokens_to_add = [t for t in sentinel_tokens if t not in existing_special_tokens]
        if tokens_to_add:
            print(f"Adding {len(tokens_to_add)} sentinel tokens to tokenizer")
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    model = ModernT5ForConditionalGeneration.from_pretrained(args.model_path)

    # Resize model embeddings if new tokens were added.
    if model.config.vocab_size < len(tokenizer):
        print(
            f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)} "
            "to accommodate tokenizer's special tokens."
        )
        model.resize_token_embeddings(len(tokenizer))

    # Create data collator
    print("Creating UL2MoDCollator")
    data_collator = UL2MoDCollator(
        tokenizer=tokenizer,
        max_seq_length=1024,
    )

    effective_bs = 512
    bs = 128
    grad_accum_steps = effective_bs // bs

    # Define training arguments first
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=1,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum_steps,
        save_steps=250,
        save_total_limit=3,  # Only keep the 3 most recent checkpoints
        warmup_steps=600,
        weight_decay=0.1,
        max_grad_norm=1.0,
        optim='adamw_torch_fused',
        lr_scheduler_type='cosine',
        remove_unused_columns=False,  # Required for custom collator
        report_to="wandb",
        bf16=True,  # Use mixed precision training if available
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train model
    print("Starting training")
    trainer.train()

    # Save final model and tokenizer
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training completed successfully")

if __name__ == "__main__":
    main()
