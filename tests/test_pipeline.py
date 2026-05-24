"""
Basic unit tests — run with: pytest tests/
These do NOT require GPU or a trained model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import (
    LABEL2ID,
    ID2LABEL,
    prepare_splits,
    load_financial_phrasebank,
)
from inference import LABEL_EMOJI, LABEL_COLOR


# ──────────────────────────────────────────────
# Data tests
# ──────────────────────────────────────────────

def test_label_maps_consistent():
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label


def test_load_data_returns_dataframe():
    df = load_financial_phrasebank("data_nonexistent_path_fallback")
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df) > 0


def test_load_data_valid_labels():
    df = load_financial_phrasebank("data_nonexistent_path_fallback")
    assert set(df["label"].unique()).issubset(set(LABEL2ID.keys()))


def test_prepare_splits_sizes():
    df = load_financial_phrasebank("data_nonexistent_path_fallback")
    train, val, test = prepare_splits(df)
    total = len(train) + len(val) + len(test)
    assert total == len(df.dropna(subset=["text", "label"]))
    # Train should be largest split
    assert len(train) > len(val)
    assert len(train) > len(test)


def test_prepare_splits_no_overlap():
    df = load_financial_phrasebank("data_nonexistent_path_fallback")
    train, val, test = prepare_splits(df)
    train_texts = set(train["text"].tolist())
    val_texts = set(val["text"].tolist())
    test_texts = set(test["text"].tolist())
    # Should have no exact duplicates across splits (given the synthetic data)
    # We check at least that the splits are disjoint in index space
    assert set(train.index).isdisjoint(set(val.index))
    assert set(train.index).isdisjoint(set(test.index))
    assert set(val.index).isdisjoint(set(test.index))


# ──────────────────────────────────────────────
# Inference helpers tests
# ──────────────────────────────────────────────

def test_label_emoji_all_labels():
    for label in LABEL2ID:
        assert label in LABEL_EMOJI
        assert label in LABEL_COLOR


def test_label_colors_are_hex():
    for label, color in LABEL_COLOR.items():
        assert color.startswith("#"), f"{label} color {color!r} is not hex"
        assert len(color) == 7


# ──────────────────────────────────────────────
# Baseline model tests (require scikit-learn only)
# ──────────────────────────────────────────────

def test_tfidf_lr_pipeline_runs():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    texts = [
        "Record profits beat analyst expectations",
        "The firm will hold its AGM next month",
        "Revenue fell short amid supply chain issues",
        "Strong quarterly growth driven by cloud segment",
        "No material changes to dividend policy announced",
        "Profit warning issued after sales decline",
    ]
    labels = [2, 1, 0, 2, 1, 0]

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, labels)
    preds = clf.predict(X)

    assert len(preds) == len(labels)
    # Training accuracy should be high on this tiny set
    acc = np.mean(preds == labels)
    assert acc > 0.5


# ──────────────────────────────────────────────
# Tokenizer smoke test (downloads if not cached)
# ──────────────────────────────────────────────

def test_tokenizer_loads():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    out = tokenizer("Test financial sentence.", return_tensors="pt")
    assert "input_ids" in out
    assert out["input_ids"].shape[1] > 0
