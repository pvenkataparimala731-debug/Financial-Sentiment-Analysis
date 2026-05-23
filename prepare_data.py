"""
Download and prepare the Financial PhraseBank dataset.

Usage:
    python src/prepare_data.py

The script downloads Financial PhraseBank (sentences with 50%+ agreement)
from the commonly used Hugging Face mirror and saves it as
data/financial_phrasebank.csv  with columns: text, label
"""

import os
import sys
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    print("Please install the datasets library: pip install datasets")
    sys.exit(1)


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    print("Downloading Financial PhraseBank (sentences_50agree split)...")
    try:
        ds = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
        train_split = ds["train"]

        df = pd.DataFrame({
            "text": train_split["sentence"],
            "label": [LABEL_MAP[l] for l in train_split["label"]],
        })

        out_path = out_dir / "financial_phrasebank.csv"
        df.to_csv(out_path, index=False)

        print(f"Saved {len(df)} examples to {out_path}")
        print(df["label"].value_counts().to_string())
    except Exception as e:
        print(f"Download failed: {e}")
        print(
            "You can manually download the dataset from:\n"
            "https://huggingface.co/datasets/financial_phrasebank\n"
            "and save it as data/financial_phrasebank.csv with columns: text, label"
        )
        print("\nFalling back to synthetic data — training will still work.")


if __name__ == "__main__":
    main()
