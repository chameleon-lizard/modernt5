#!/usr/bin/env python
"""
build_ul2_tokenizer.py

Train a SentencePiece tokenizer for ModernT5 with UL2 mixture-of-denoisers tokens.

Requirements:
  pip install datasets sentencepiece transformers

Usage:
  python build_ul2_tokenizer.py \
        --data_dir ./combined_dataset \
        --text_column text \
        --vocab_size 33280 \
        --out_prefix modernt5_sp \
        --tokenizer_dir ./modernt5_tokenizer
"""
import argparse
from pathlib import Path
import sentencepiece as spm
from datasets import load_from_disk
from transformers import T5Tokenizer

# --------------------------------------------------------------------------- #
# 1  Special-token inventory (UL2)                                            #
# --------------------------------------------------------------------------- #
UL2_PREFIXES = ["[NLU]", "[S2S]", "[NLG]"]                # R, S, X denoisers
SENTINELS    = [f"<extra_id_{i}>" for i in range(100)]    # <extra_id_0> … 99
SPECIAL_TOKENS = ["<pad>", "</s>", "<unk>"] + UL2_PREFIXES + SENTINELS
# --------------------------------------------------------------------------- #


def sentence_iterator(data_dir: Path, text_column: str):
    """Yield raw text lines from every JSON file under data_dir."""
    ds = load_from_disk(
        str(data_dir),
    )

    for row in ds:
        txt = row.get(text_column, "").strip()
        if txt:
            yield txt


def train_spm(data_dir: Path,
              text_column: str,
              vocab_size: int,
              model_prefix: str):
    """Train SentencePiece on a streamed HF dataset."""
    print(f"→ Training SentencePiece ({vocab_size} tokens)…")
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator(data_dir, text_column),
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=1.0,           # fine for English-heavy corpora
        unk_piece="<unk>",  unk_id=0,
        pad_piece="<pad>",  pad_id=1,
        eos_piece="</s>",   eos_id=2,
        bos_id=-1,                          # T5/UL2 never uses <bos>
        control_symbols=",".join(SPECIAL_TOKENS[3:]),  # all except pad/unk/eos
        train_extremely_large_corpus=True
    )
    print("✓ SentencePiece model written:", f"{model_prefix}.model")


def wrap_hf_tokenizer(sp_model_prefix: str,
                      tokenizer_dir: Path):
    """Create & save a HF T5Tokenizer around the SPM model."""
    print("→ Building Hugging Face wrapper…")
    tok = T5Tokenizer(vocab_file=f"{sp_model_prefix}.model",
                       extra_ids=100)               # <extra_id_0-99>
    tok.add_special_tokens({"additional_special_tokens": UL2_PREFIXES})
    assert len(tok) == 33_280, "Tokenizer size mismatch!"
    tok.save_pretrained(tokenizer_dir)
    print("✓ HF tokenizer saved to", tokenizer_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("./combined_dataset"))
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--vocab_size", type=int, default=33_280)
    parser.add_argument("--out_prefix", type=str, default="modernt5_sp")
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("./modernt5_tokenizer"))
    args = parser.parse_args()

    train_spm(args.data_dir, args.text_column,
              args.vocab_size, args.out_prefix)
    wrap_hf_tokenizer(args.out_prefix, args.tokenizer_dir)


if __name__ == "__main__":
    main()

