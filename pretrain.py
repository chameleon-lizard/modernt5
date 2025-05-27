import os
import logging
import argparse
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    set_seed,
)

from modeling_modernt5 import ModernT5ForConditionalGeneration
from collator import UL2MoDCollator
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train ModernT5ForConditionalGeneration model")
    parser.add_argument("--model_path", type=str, default="modernt5_checkpoint", 
                        help="Path to the model checkpoint directory")
    parser.add_argument("--dataset_dir", type=str, default="final_pretrain_mix", 
                        help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, 
                        help="Batch size per GPU/TPU")
    parser.add_argument("--learning_rate", type=float, default=5e-4, 
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
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
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = ModernT5ForConditionalGeneration.from_pretrained(args.model_path)
    
    # Create data collator
    logger.info("Creating UL2MoDCollator")
    data_collator = UL2MoDCollator(
        tokenizer=tokenizer,
        max_seq_length=2048,
    )

    effective_bs = 512
    bs = 1
    grad_accum_steps = effective_bs // bs
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        learning_rate=5.5e-4,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=1,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum_steps,
        save_steps=250,
        save_total_limit=3,  # Only keep the 3 most recent checkpoints
        warmup_steps=40,
        weight_decay=0.1,
        optim='paged_adamw_8bit',
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

    run = wandb.init(
        name="modernt5_pretrain_test_lr",
        project="ModernT5",
        tags=["pretrain"],
    )
    wandb.watch(model, log=None, log_freq=10)

    # Train model
    logger.info("Starting training")
    trainer.train()
    
    run.finish()

    # Save final model and tokenizer
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()

