"""
Inference helpers — load a trained model (or fall back to a pretrained
sentiment model) and run predictions.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Emoji helpers
LABEL_EMOJI = {"positive": "📈", "neutral": "➖", "negative": "📉"}
LABEL_COLOR = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}


class SentimentPredictor:
    """
    Wraps a fine-tuned (or pretrained) HuggingFace model for financial
    sentiment inference.  Falls back gracefully when no local model exists.
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model_path and Path(model_path).exists():
            print(f"Loading fine-tuned model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(device)
            self.model.eval()
            self._mode = "finetuned"
        else:
            # Fall back to a general-purpose pretrained model
            fallback = "ProsusAI/finbert"
            print(f"No local model found — loading pretrained {fallback}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model = AutoModelForSequenceClassification.from_pretrained(fallback)
            self.model.to(device)
            self.model.eval()
            self._mode = "pretrained"

    @property
    def mode(self) -> str:
        return self._mode

    def predict(self, texts: List[str], max_length: int = 128) -> List[Dict[str, Any]]:
        """
        Run inference on a list of texts.

        Returns a list of dicts with keys:
            text, label, confidence, probabilities
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i, text in enumerate(texts):
            pred_id = int(np.argmax(probs[i]))
            label = ID2LABEL[pred_id]
            results.append(
                {
                    "text": text,
                    "label": label,
                    "confidence": float(probs[i][pred_id]),
                    "probabilities": {
                        "negative": float(probs[i][0]),
                        "neutral": float(probs[i][1]),
                        "positive": float(probs[i][2]),
                    },
                    "emoji": LABEL_EMOJI[label],
                    "color": LABEL_COLOR[label],
                }
            )
        return results

    def predict_single(self, text: str) -> Dict[str, Any]:
        return self.predict([text])[0]


class BaselinePredictor:
    """
    Wraps the TF-IDF + Logistic Regression baseline model.
    """

    def __init__(self, model_dir: str):
        vec_path = Path(model_dir) / "tfidf_vectorizer.joblib"
        clf_path = Path(model_dir) / "lr_classifier.joblib"

        if not vec_path.exists() or not clf_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found in {model_dir}. Run train.py first."
            )

        self.vectorizer = joblib.load(vec_path)
        self.clf = joblib.load(clf_path)

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)
        proba = self.clf.predict_proba(X)

        results = []
        for i, text in enumerate(texts):
            label = ID2LABEL[preds[i]]
            results.append(
                {
                    "text": text,
                    "label": label,
                    "confidence": float(proba[i][preds[i]]),
                    "probabilities": {
                        "negative": float(proba[i][0]),
                        "neutral": float(proba[i][1]),
                        "positive": float(proba[i][2]),
                    },
                    "emoji": LABEL_EMOJI[label],
                    "color": LABEL_COLOR[label],
                }
            )
        return results

    def predict_single(self, text: str) -> Dict[str, Any]:
        return self.predict([text])[0]


def load_results(results_path: str) -> dict | None:
    """Load training results JSON if available."""
    p = Path(results_path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None
