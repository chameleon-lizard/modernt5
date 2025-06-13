import os
import logging
import argparse
import json
import shutil
import tempfile
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize a dataset using a Hugging Face tokenizer")
    parser.add_argument("--dataset_dir", type=str, default="final_pretrain_mix",
                        help="Path to the directory containing the raw dataset.")
    parser.add_argument("--tokenizer_path", type=str, default="modernt5_tokenizer",
                        help="Path to the directory containing the tokenizer.")
    parser.add_argument("--output_dir", type=str, default="final_pretrain_mix_tokenized",
                        help="Directory to save the tokenized dataset.")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--processing_batch_size", type=int, default=1000,
                        help="Batch size for processing and tokenizing the dataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()


def main():
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
    # Assuming the dataset has a 'text' column based on typical raw datasets
    try:
        dataset = load_from_disk(args.dataset_dir)
        if "text" not in dataset.column_names:
             raise ValueError(f"Dataset must contain a 'text' column. Found: {dataset.column_names}")
    except Exception as e:
        logger.error(f"Failed to load dataset or 'text' column not found: {e}")
        return


    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
        return

    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory for intermediate files: {temp_dir}")

    jsonl_paths = []
    total_tokens = 0

    logger.info(f"Tokenizing and processing dataset in batches of {args.processing_batch_size}...")

    for i in range(0, len(dataset), args.processing_batch_size):
        batch = dataset[i:i+args.processing_batch_size]
        
        # Tokenize the batch
        tokenized_batch = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False, # Collator handles padding later
        )

        # Update total token count
        total_tokens += sum(len(ids) for ids in tokenized_batch['input_ids'])

        # Prepare records for JSONL format
        records = []
        num_examples_in_batch = len(tokenized_batch['input_ids'])
        for j in range(num_examples_in_batch):
            records.append({
                'input_ids': tokenized_batch['input_ids'][j],
                'attention_mask': tokenized_batch['attention_mask'][j],
            })

        # Write batch to a temporary JSONL file
        batch_num = i // args.processing_batch_size
        jsonl_path = os.path.join(temp_dir, f"batch_{batch_num}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        jsonl_paths.append(jsonl_path)

        if batch_num % 10 == 0:
            logger.info(f"Processed {i + num_examples_in_batch} / {len(dataset)} examples...")

    logger.info("Finished processing all batches.")
    
    # Log and save total token count
    logger.info(f"The tokenized dataset contains {total_tokens:,} tokens.")
    info_file_path = os.path.join(args.output_dir, "dataset_info.md")
    with open(info_file_path, "w") as f:
        f.write(f"# Dataset Info\n\n")
        f.write(f"Total number of tokens: {total_tokens:,}\n")
    logger.info(f"Dataset info saved to {info_file_path}")

    # Load the processed data from JSONL files into a single dataset
    logger.info(f"Loading tokenized data from {len(jsonl_paths)} temporary files...")
    # We assume the data is for the 'train' split.
    tokenized_dataset = load_dataset("json", data_files=jsonl_paths, split="train")

    # Ensure the 'labels' column is not present yet, as the collator creates it
    if 'labels' in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(['labels'])
        logger.warning("Removed existing 'labels' column as it will be generated by the collator.")

    # Save the final tokenized dataset
    logger.info(f"Saving final tokenized dataset to {args.output_dir}")
    tokenized_dataset.save_to_disk(args.output_dir)

    # Clean up temporary directory
    logger.info(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)

    logger.info("Tokenization and saving completed successfully.")


if __name__ == "__main__":
    main()
