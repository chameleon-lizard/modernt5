import argparse
import json
from datasets import load_from_disk
from transformers import AutoTokenizer
from collator import UL2MoDCollator


def decode_ids(tokenizer, ids):
    """Decode token ids to string, filtering out padding and negative values."""
    valid_ids = [i for i in ids if i >= 0 and i != tokenizer.pad_token_id]
    # For ids beyond the tokenizer vocab, map to unk token to avoid errors
    mapped_ids = [i if i < tokenizer.vocab_size else tokenizer.unk_token_id for i in valid_ids]
    return tokenizer.decode(mapped_ids, skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate collated examples for human inspection")
    parser.add_argument("--dataset_dir", type=str, default="final_pretrain_mix_tokenized",
                        help="Path to the tokenized dataset directory.")
    parser.add_argument("--tokenizer_path", type=str, default="modernt5_from_rumodernbert",
                        help="Path to the tokenizer or model directory.")
    parser.add_argument("--output_file", type=str, default="collator_examples.json",
                        help="Destination JSON file for generated examples.")
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Number of examples to generate.")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for the collator.")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    collator = UL2MoDCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    examples = []
    limit = min(args.num_examples, len(dataset))
    for i in range(limit):
        batch = collator([dataset[i]])
        input_ids = batch["input_ids"][0].tolist()
        attention_mask = batch["attention_mask"][0].tolist()
        labels = batch["labels"][0].tolist()

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_text": decode_ids(tokenizer, input_ids),
            "label_text": decode_ids(tokenizer, [j for j in labels if j != -100]),
        }
        examples.append(example)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
