import argparse
import os
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm.auto import tqdm

# (dataset_name, split, text_column, gigabytes_to_take)
DATASETS = [
    ("deepvk/cultura_ru_edu", "train", "text", 30),
    ("HuggingFaceFW/fineweb-edu", "train", "text", 30),
    ("angie-chen55/python-github-code", "train", "code", 10),
]

BYTES_IN_GB = 1024 ** 3

def take_n_gigabytes(name: str, split: str, column: str, n_gb: float) -> Dataset:
    """Stream *split* of *name* and return a Dataset with up to *n_gb* GB from *column*."""
    budget = int(n_gb * BYTES_IN_GB)
    rng_stream = load_dataset(name, split=split, streaming=True)

    records = []
    total = 0
    for sample in tqdm(rng_stream, desc=f"Reading {name}"):
        value = sample[column]
        if value is None:
            continue
        size = len(value.encode("utf-8"))
        if total + size > budget:
            break
        records.append({"text": value})  # unify all columns to `text`
        total += size

    print(f"→ {len(records)} records | {total / BYTES_IN_GB:.2f} GB collected from {name}")
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine slice of multiple HF datasets")
    parser.add_argument(
        "--output_repo",
        type=str,
        help="Destination HF repo (username/repo). If omitted, dataset is saved to ./combined_dataset instead.",
    )
    args = parser.parse_args()

    parts = [take_n_gigabytes(*cfg) for cfg in DATASETS]
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

