import argparse
import json
import os
import tempfile
from typing import Optional

from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm.auto import tqdm


BYTES_IN_GB = 1024**3


def take_n_gigabytes(name: str, split: str, column: str, n_gb: float, temp_dir: Optional[str] = None) -> Dataset:
    """Stream *split* of *name* and return a Dataset with up to *n_gb* GB from *column*."""
    budget = int(n_gb * BYTES_IN_GB)
    rng_stream = load_dataset(name, split=split, streaming=True)

    # To avoid OOM, we'll write to a temporary file on disk instead of keeping records in memory.
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", suffix=".jsonl", dir=temp_dir
    ) as f:
        temp_filename = f.name
        total_bytes = 0
        num_records = 0
        for sample in tqdm(rng_stream, desc=f"Reading {name}"):
            value = sample[column]
            if value is None:
                continue

            record = {"text": value}  # unify all columns to `text`
            record_str = json.dumps(record, ensure_ascii=False)
            # Add 1 for the newline character to accurately track file size
            record_size = len(record_str.encode("utf-8")) + 1

            if total_bytes + record_size > budget:
                break

            f.write(record_str + "\n")
            total_bytes += record_size
            num_records += 1

    print(f"→ {num_records} records | {total_bytes / BYTES_IN_GB:.2f} GB collected from {name}")

    # Load the dataset from the temporary JSONL file. This will be memory-mapped.
    dataset = load_dataset("json", data_files=temp_filename, split="train")

    # Clean up the temporary file
    os.remove(temp_filename)

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine slice of multiple HF datasets")
    parser.add_argument(
        "--output_repo",
        type=str,
        help="Destination HF repo (username/repo). If omitted, dataset is saved to ./combined_dataset instead.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="./cache",
        help="Directory for temporary files. Defaults to ./cache",
    )
    parser.add_argument(
        "--en",
        type=int,
        default=20,
        help="Gigabytes to take from English datasets (fineweb-edu).",
    )
    parser.add_argument(
        "--ru",
        type=int,
        default=20,
        help="Gigabytes to take from Russian datasets (cultura_ru_edu).",
    )
    parser.add_argument(
        "--code",
        type=int,
        default=6,
        help="Gigabytes to take from code datasets (python-github-code).",
    )
    args = parser.parse_args()

    DATASETS = [
        ("deepvk/cultura_ru_edu", "train", "text", args.ru),
        ("HuggingFaceFW/fineweb-edu", "train", "text", args.en),
        ("angie-chen51/python-github-code", "train", "code", args.code),
    ]

    os.makedirs(args.temp_dir, exist_ok=True)

    parts = [take_n_gigabytes(*cfg, temp_dir=args.temp_dir) for cfg in DATASETS]
    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=42)

    if args.output_repo:
        combined.push_to_hub(args.output_repo)
        print(f"✅ Pushed combined dataset to {args.output_repo}")
    else:
        out_dir = "final_pretrain_mix"
        combined.save_to_disk(out_dir)
        print(f"💾 Saved combined dataset to {out_dir}")


if __name__ == "__main__":
    main()
