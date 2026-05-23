"""
Fine-tune DistilBERT for financial sentiment analysis.
Compares against a TF-IDF + Logistic Regression baseline.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import evaluate
import joblib
from tqdm import tqdm

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_financial_phrasebank(data_dir: str) -> pd.DataFrame:
    """
    Load Financial PhraseBank dataset.
    Falls back to a small synthetic sample so the pipeline always runs.
    """
    fp = Path(data_dir) / "financial_phrasebank.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        print(f"Loaded {len(df)} rows from {fp}")
        return df

    print("Dataset not found — generating synthetic sample for demo purposes.")
    positive_examples = [
        "The company reported record earnings this quarter, beating analyst expectations.",
        "Revenue surged 40% year-over-year driven by strong product demand.",
        "The firm announced a major acquisition that is expected to boost market share.",
        "Operating margins expanded significantly thanks to cost-cutting initiatives.",
        "Investors cheered the dividend increase announced after the earnings call.",
        "The stock hit an all-time high after positive clinical trial results.",
        "Management raised full-year guidance, citing robust consumer demand.",
        "The partnership deal unlocked new markets and accelerated growth.",
        "Free cash flow reached its highest level in company history.",
        "The board approved a $2 billion share buyback program.",
    ]
    negative_examples = [
        "The company missed earnings estimates and lowered its forward guidance.",
        "Regulatory investigations weighed heavily on the stock price today.",
        "Declining sales in key markets led to the company's worst quarter in years.",
        "The CEO resigned unexpectedly amid mounting pressure from shareholders.",
        "Debt levels have risen to alarming proportions following the failed merger.",
        "Supply chain disruptions caused production to halt at two major facilities.",
        "The firm was forced to write down $500 million in goodwill assets.",
        "Customer churn accelerated as competitors undercut pricing aggressively.",
        "Layoffs of 10% of the workforce were announced to cut costs.",
        "The product recall will cost an estimated $300 million in damages.",
    ]
    neutral_examples = [
        "The company will hold its annual shareholder meeting on June 15.",
        "Management noted that market conditions remain unchanged from last quarter.",
        "The firm operates in 24 countries and employs approximately 15,000 people.",
        "The board will review the dividend policy at its next scheduled meeting.",
        "The acquisition process is ongoing and no timeline has been confirmed.",
        "Quarterly results will be released before market open on Thursday.",
        "The company has not issued any formal comment on the analyst report.",
        "Trading volume was in line with the 30-day average for the stock.",
        "The CFO will present at the industry conference next month.",
        "The company completed the previously announced restructuring program.",
    ]

    texts = positive_examples + negative_examples + neutral_examples
    labels = (
        ["positive"] * len(positive_examples)
        + ["negative"] * len(negative_examples)
        + ["neutral"] * len(neutral_examples)
    )

    # Augment to a more realistic size
    rng = np.random.default_rng(42)
    indices = rng.choice(len(texts), size=600, replace=True)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    df = pd.DataFrame({"text": texts, "label": labels})
    return df


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def prepare_splits(df: pd.DataFrame, seed: int = 42):
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label_id"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label_id"])

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128):
    dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": acc,
    }


# ──────────────────────────────────────────────
# Baseline: TF-IDF + Logistic Regression
# ──────────────────────────────────────────────

def train_baseline(train_df, val_df, test_df, output_dir: str):
    print("\n" + "=" * 50)
    print("Training Baseline: TF-IDF + Logistic Regression")
    print("=" * 50)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30_000,
        sublinear_tf=True,
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X_train, train_df["label_id"])

    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test)

    val_f1 = f1_score(val_df["label_id"], val_preds, average="macro")
    test_f1 = f1_score(test_df["label_id"], test_preds, average="macro")
    test_acc = accuracy_score(test_df["label_id"], test_preds)

    print(f"Validation F1 (macro): {val_f1:.4f}")
    print(f"Test F1 (macro):       {test_f1:.4f}")
    print(f"Test Accuracy:         {test_acc:.4f}")
    print("\nDetailed Test Report:")
    print(
        classification_report(
            test_df["label_id"],
            test_preds,
            target_names=list(LABEL2ID.keys()),
        )
    )

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(output_dir, "lr_classifier.joblib"))

    results = {
        "val_f1_macro": val_f1,
        "test_f1_macro": test_f1,
        "test_accuracy": test_acc,
        "test_f1_weighted": float(f1_score(test_df["label_id"], test_preds, average="weighted")),
        "confusion_matrix": confusion_matrix(test_df["label_id"], test_preds).tolist(),
    }

    with open(os.path.join(output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline artifacts saved to {output_dir}")
    return results


# ──────────────────────────────────────────────
# BERT Fine-tuning
# ──────────────────────────────────────────────

def train_bert(
    train_df,
    val_df,
    test_df,
    model_name: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    print("\n" + "=" * 50)
    print(f"Fine-tuning: {model_name}")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize_dataset(val_df, tokenizer)
    test_dataset = tokenize_dataset(test_df, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Evaluate on test set
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids

    test_f1_macro = f1_score(test_labels, test_preds, average="macro")
    test_f1_weighted = f1_score(test_labels, test_preds, average="weighted")
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"\nTest F1 (macro):    {test_f1_macro:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")
    print(f"Test Accuracy:      {test_acc:.4f}")
    print("\nDetailed Test Report:")
    print(classification_report(test_labels, test_preds, target_names=list(LABEL2ID.keys())))

    # Save model and tokenizer
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    results = {
        "model_name": model_name,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "test_accuracy": test_acc,
        "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
        "epochs_trained": trainer.state.epoch,
        "best_val_f1": trainer.state.best_metric,
    }

    with open(os.path.join(output_dir, "bert_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBERT model saved to {best_model_dir}")
    return results


# ──────────────────────────────────────────────
# Comparison summary
# ──────────────────────────────────────────────

def print_comparison(baseline_results: dict, bert_results: dict):
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>12} {'DistilBERT':>12} {'Delta':>10}")
    print("-" * 60)

    for key in ["test_f1_macro", "test_f1_weighted", "test_accuracy"]:
        b = baseline_results[key]
        bert = bert_results[key]
        delta = bert - b
        sign = "+" if delta >= 0 else ""
        label = key.replace("test_", "").replace("_", " ").title()
        print(f"{label:<25} {b:>12.4f} {bert:>12.4f} {sign}{delta:>9.4f}")

    f1_improvement = (
        (bert_results["test_f1_macro"] - baseline_results["test_f1_macro"])
        / baseline_results["test_f1_macro"]
        * 100
    )
    print("-" * 60)
    print(f"\nF1 (macro) improvement: {f1_improvement:+.1f}%")
    print("=" * 60)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Financial Sentiment Analysis Training")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Where to save models")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_bert", action="store_true", help="Only run baseline (faster for testing)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and split data
    df = load_financial_phrasebank(args.data_dir)
    train_df, val_df, test_df = prepare_splits(df, seed=args.seed)

    # Save splits for reproducibility
    os.makedirs(args.data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.data_dir, "test.csv"), index=False)

    # Baseline
    baseline_dir = os.path.join(args.output_dir, "baseline")
    baseline_results = train_baseline(train_df, val_df, test_df, baseline_dir)

    if not args.skip_bert:
        bert_dir = os.path.join(args.output_dir, "bert")
        bert_results = train_bert(
            train_df, val_df, test_df,
            model_name=args.model_name,
            output_dir=bert_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print_comparison(baseline_results, bert_results)

        # Save combined results for the Streamlit app
        combined = {
            "baseline": baseline_results,
            "bert": bert_results,
            "trained_at": datetime.now().isoformat(),
            "model_path": os.path.join(bert_dir, "best_model"),
        }
        with open(os.path.join(args.output_dir, "combined_results.json"), "w") as f:
            json.dump(combined, f, indent=2)
    else:
        print("\nSkipped BERT training (--skip_bert flag set).")


if __name__ == "__main__":
    main()
