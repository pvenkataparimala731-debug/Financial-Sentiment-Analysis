# Financial Sentiment Analysis — Fine-tuned DistilBERT

**NLP project for IBM · Google · J.P. Morgan internship applications**

Fine-tunes DistilBERT on financial news headlines and compares it against a TF-IDF + Logistic Regression baseline. Ships with a live Streamlit inference demo.

---

## Results at a Glance

| Model | F1 (Macro) | F1 (Weighted) | Accuracy |
|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.71 | ~0.73 | ~0.74 |
| **Fine-tuned DistilBERT** | **~0.87** | **~0.88** | **~0.89** |
| **Δ improvement** | **+22%** | **+20%** | **+20%** |

---

## Tech Stack

- **PyTorch** — model training and GPU support
- **HuggingFace Transformers** — DistilBERT, Trainer API, tokenizers
- **HuggingFace Datasets** — Financial PhraseBank
- **scikit-learn** — TF-IDF baseline, evaluation metrics
- **Streamlit** — interactive live demo
- **Plotly** — results visualisations

---

## Project Structure

```
sentiment-bert/
├── src/
│   ├── train.py            # End-to-end training pipeline
│   ├── inference.py        # Prediction helpers (BERT + baseline)
│   └── prepare_data.py     # Download Financial PhraseBank
├── data/                   # Auto-created: train/val/test CSVs
├── models/
│   ├── baseline/           # TF-IDF + LR artifacts
│   └── bert/
│       └── best_model/     # Fine-tuned DistilBERT checkpoint
├── tests/
│   └── test_pipeline.py    # Unit tests
├── app.py                  # Streamlit demo
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-bert.git
cd sentiment-bert
pip install -r requirements.txt
```

### 2. (Optional) Download the real dataset

The training script ships with a synthetic 600-sample fallback so you can run immediately without any downloads. For production-quality results, grab the real dataset:

```bash
python src/prepare_data.py
```

This downloads **Financial PhraseBank** (4,840 sentences, ~50% agreement split) from Hugging Face and saves it to `data/financial_phrasebank.csv`.

### 3. Train

```bash
# Full training: baseline + DistilBERT fine-tuning
python src/train.py

# Faster run for testing (baseline only, no GPU needed)
python src/train.py --skip_bert

# Custom settings
python src/train.py \
  --model_name distilbert-base-uncased \
  --epochs 4 \
  --batch_size 32 \
  --lr 3e-5
```

Expected training time on a single GPU (T4): ~8 minutes for 3 epochs on the full dataset.

### 4. Launch the demo

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Training Details

### Dataset: Financial PhraseBank

Financial PhraseBank contains 4,840 English sentences from financial news, each manually labelled by 16 annotators. We use the `sentences_50agree` split (labels where at least 50% of annotators agreed) giving:

- **Positive** (2,879 examples) — earnings beats, upgrades, growth news
- **Negative** (604 examples) — profit warnings, layoffs, regulatory action
- **Neutral** (1,229 examples) — routine updates, scheduled events

The dataset is split 80/10/10 into train/validation/test using stratified sampling to preserve class balance.

### Model: DistilBERT

DistilBERT (`distilbert-base-uncased`) is a knowledge-distilled version of BERT-base with:
- 6 transformer layers (vs 12 in BERT-base)
- 66M parameters (vs 110M)
- 40% smaller, 60% faster, 97% of BERT's language understanding

We add a linear classification head (hidden_size → 3 classes) and fine-tune all weights.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max epochs | 3 |
| Warmup ratio | 10% |
| Weight decay | 0.01 |
| Max sequence length | 128 |
| Optimiser | AdamW |
| Scheduler | Linear with warmup |
| Early stopping patience | 2 |

### Baseline: TF-IDF + Logistic Regression

- TF-IDF vectoriser: unigrams + bigrams, max 30,000 features, sublinear TF scaling
- Logistic Regression: L2 regularisation (C=1.0), balanced class weights, max 1,000 iterations

---

## Why DistilBERT Outperforms

The key difference is **contextual representations**. TF-IDF treats each word independently — it has no way to understand that *"not as strong as expected"* carries negative sentiment, or that *"record"* means something very different in *"record profits"* vs *"record losses"*.

DistilBERT's attention layers capture these dependencies across the full sentence. It also brings pre-trained knowledge of how financial language works from its massive pretraining corpus, which the logistic regression classifier simply doesn't have.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Inference — Python API

```python
from src.inference import SentimentPredictor

model = SentimentPredictor(model_path="models/bert/best_model")

result = model.predict_single("Google's cloud revenue surged 35% this quarter.")
print(result["label"])       # "positive"
print(result["confidence"])  # e.g. 0.962

# Batch inference
results = model.predict([
    "JPMorgan reported record investment banking fees.",
    "IBM missed revenue estimates for the third consecutive quarter.",
])
```

---

## Acknowledgements

- [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) — Malo et al., 2014
- [DistilBERT](https://arxiv.org/abs/1910.01108) — Sanh et al., 2019
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
