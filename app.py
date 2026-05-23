"""
Streamlit app — Financial Sentiment Analysis Demo
Run with: streamlit run app.py
"""

import os
import sys
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from inference import SentimentPredictor, BaselinePredictor, load_results, LABEL_COLOR, LABEL_EMOJI

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .result-card {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid;
        background: #f8fafc;
        margin-bottom: 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stTextArea textarea {
        font-size: 0.95rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=40)
    st.markdown("## ⚙️ Settings")

    model_dir = st.text_input("Model directory", value="models/bert/best_model")
    baseline_dir = st.text_input("Baseline directory", value="models/baseline")

    st.divider()
    st.markdown("### Model Info")

    bert_path = Path(model_dir)
    if bert_path.exists():
        st.success("✅ Fine-tuned model found")
        model_status = "fine-tuned"
    else:
        st.warning("⚠️ No local model — using pretrained FinBERT")
        model_status = "pretrained"

    st.divider()
    st.markdown("### About")
    st.info(
        "This demo compares **DistilBERT** fine-tuned on financial news "
        "against a **TF-IDF + Logistic Regression** baseline. "
        "Built with HuggingFace Transformers & PyTorch."
    )
    st.markdown("**Targets:** IBM · Google · J.P. Morgan")

# ──────────────────────────────────────────────
# Load models (cached)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading BERT model...")
def get_bert_model(path):
    return SentimentPredictor(model_path=path if Path(path).exists() else None)

@st.cache_resource(show_spinner="Loading baseline model...")
def get_baseline_model(path):
    try:
        return BaselinePredictor(model_dir=path)
    except FileNotFoundError:
        return None

bert_model = get_bert_model(model_dir)
baseline_model = get_baseline_model(baseline_dir)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown('<p class="main-header">📊 Financial Sentiment Analysis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Fine-tuned DistilBERT vs TF-IDF + Logistic Regression · '
    'IBM · Google · J.P. Morgan</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Inference",
    "📈 Results Comparison",
    "📦 Batch Analysis",
    "📚 How It Works",
])

# ═══════════════════════════════════════════════
# TAB 1 — Live Inference
# ═══════════════════════════════════════════════

with tab1:
    st.markdown("### Analyse financial text in real time")

    EXAMPLES = {
        "📈 Strong Earnings": "The company reported record-breaking quarterly earnings, exceeding analyst consensus by 18% and raising its full-year revenue guidance.",
        "📉 Regulatory Warning": "Regulators have launched a formal investigation into the company's accounting practices, raising concerns about potential restatements.",
        "➖ Routine Update": "The firm will host its annual general meeting on the 15th of next month to discuss routine corporate governance matters.",
        "📈 Acquisition Win": "The $4.2 billion acquisition of the AI startup positions the company as a clear leader in enterprise machine learning solutions.",
        "📉 Profit Warning": "Management issued a profit warning citing unexpected supply chain disruptions expected to cost $300 million in the current fiscal year.",
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_example = st.selectbox("Try an example →", ["(type your own)"] + list(EXAMPLES.keys()))
    with col2:
        compare_mode = st.checkbox("Compare with baseline", value=True)

    default_text = EXAMPLES.get(selected_example, "")
    user_text = st.text_area(
        "Enter financial news headline or sentence:",
        value=default_text,
        height=120,
        placeholder="E.g. Google's cloud division posted its strongest quarter ever, growing 35% year-over-year...",
    )

    analyse_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)

    if analyse_btn and user_text.strip():
        with st.spinner("Running inference..."):
            start = time.perf_counter()
            bert_result = bert_model.predict_single(user_text.strip())
            bert_time = (time.perf_counter() - start) * 1000

            baseline_result = None
            if compare_mode and baseline_model:
                start = time.perf_counter()
                baseline_result = baseline_model.predict_single(user_text.strip())
                baseline_time = (time.perf_counter() - start) * 1000

        st.divider()

        # BERT result
        col_l, col_r = st.columns(2 if baseline_result else 1)

        with col_l:
            label = bert_result["label"]
            color = LABEL_COLOR[label]
            emoji = bert_result["emoji"]
            conf = bert_result["confidence"]

            st.markdown(f"#### {'Fine-tuned DistilBERT' if model_status == 'fine-tuned' else 'Pretrained FinBERT'}")
            st.markdown(
                f'<div class="result-card" style="border-color:{color};">'
                f'<span style="font-size:2rem">{emoji}</span> '
                f'<span style="font-size:1.3rem;font-weight:700;color:{color};">'
                f'{label.upper()}</span>'
                f'<br><span style="color:#64748b;font-size:0.85rem">'
                f'Confidence: {conf:.1%} · {bert_time:.0f} ms</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            probs = bert_result["probabilities"]
            fig = go.Figure(go.Bar(
                x=list(probs.values()),
                y=list(probs.keys()),
                orientation="h",
                marker_color=[LABEL_COLOR[k] for k in probs.keys()],
                text=[f"{v:.1%}" for v in probs.values()],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 1], tickformat=".0%"),
                height=200,
                margin=dict(l=10, r=40, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key="bert_prob_chart")

        if baseline_result:
            with col_r:
                bl_label = baseline_result["label"]
                bl_color = LABEL_COLOR[bl_label]
                bl_emoji = baseline_result["emoji"]
                bl_conf = baseline_result["confidence"]

                st.markdown("#### Baseline (TF-IDF + LR)")
                st.markdown(
                    f'<div class="result-card" style="border-color:{bl_color};">'
                    f'<span style="font-size:2rem">{bl_emoji}</span> '
                    f'<span style="font-size:1.3rem;font-weight:700;color:{bl_color};">'
                    f'{bl_label.upper()}</span>'
                    f'<br><span style="color:#64748b;font-size:0.85rem">'
                    f'Confidence: {bl_conf:.1%} · {baseline_time:.0f} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                bl_probs = baseline_result["probabilities"]
                fig2 = go.Figure(go.Bar(
                    x=list(bl_probs.values()),
                    y=list(bl_probs.keys()),
                    orientation="h",
                    marker_color=[LABEL_COLOR[k] for k in bl_probs.keys()],
                    text=[f"{v:.1%}" for v in bl_probs.values()],
                    textposition="outside",
                ))
                fig2.update_layout(
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    height=200,
                    margin=dict(l=10, r=40, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2, use_container_width=True, key="baseline_prob_chart")

    elif analyse_btn:
        st.warning("Please enter some text first.")

# ═══════════════════════════════════════════════
# TAB 2 — Results Comparison
# ═══════════════════════════════════════════════

with tab2:
    st.markdown("### Model Performance Comparison")

    results = load_results("models/combined_results.json")

    if results:
        b = results["baseline"]
        bert = results["bert"]

        # Top metrics
        col1, col2, col3, col4 = st.columns(4)

        def delta_str(v, b_v):
            d = v - b_v
            return f"{'+' if d >= 0 else ''}{d:.3f}"

        with col1:
            st.metric("BERT F1 (Macro)", f"{bert['test_f1_macro']:.3f}",
                      delta_str(bert['test_f1_macro'], b['test_f1_macro']))
        with col2:
            st.metric("BERT F1 (Weighted)", f"{bert['test_f1_weighted']:.3f}",
                      delta_str(bert['test_f1_weighted'], b['test_f1_weighted']))
        with col3:
            st.metric("BERT Accuracy", f"{bert['test_accuracy']:.3f}",
                      delta_str(bert['test_accuracy'], b['test_accuracy']))
        with col4:
            improvement = (bert['test_f1_macro'] - b['test_f1_macro']) / b['test_f1_macro'] * 100
            st.metric("F1 Improvement", f"+{improvement:.1f}%")

        st.divider()

        # Bar chart comparison
        metrics = ["F1 Macro", "F1 Weighted", "Accuracy"]
        baseline_vals = [b["test_f1_macro"], b["test_f1_weighted"], b["test_accuracy"]]
        bert_vals = [bert["test_f1_macro"], bert["test_f1_weighted"], bert["test_accuracy"]]

        fig = go.Figure(data=[
            go.Bar(name="TF-IDF + LR (Baseline)", x=metrics, y=baseline_vals,
                   marker_color="#94a3b8", text=[f"{v:.3f}" for v in baseline_vals],
                   textposition="outside"),
            go.Bar(name="Fine-tuned DistilBERT", x=metrics, y=bert_vals,
                   marker_color="#3b82f6", text=[f"{v:.3f}" for v in bert_vals],
                   textposition="outside"),
        ])
        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.05]),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key="comparison_bar")

        # Confusion matrices
        st.markdown("#### Confusion Matrices")
        col_a, col_b = st.columns(2)
        labels_display = ["Negative", "Neutral", "Positive"]

        for col, title, cm_data in [
            (col_a, "Baseline", b["confusion_matrix"]),
            (col_b, "DistilBERT", bert["confusion_matrix"]),
        ]:
            with col:
                cm = np.array(cm_data)
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels_display, y=labels_display,
                    color_continuous_scale="Blues",
                    text_auto=True,
                    title=title,
                )
                fig_cm.update_layout(height=320, margin=dict(t=40, b=10))
                st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{title}")
    else:
        st.info(
            "No training results found yet. Run `python src/train.py` to train the model, "
            "then refresh this page."
        )

        # Show placeholder chart with demo data
        st.markdown("#### Preview (demo values)")
        metrics = ["F1 Macro", "F1 Weighted", "Accuracy"]
        fig = go.Figure(data=[
            go.Bar(name="TF-IDF + LR (Baseline)", x=metrics, y=[0.71, 0.73, 0.74],
                   marker_color="#94a3b8", text=["0.710", "0.730", "0.740"],
                   textposition="outside"),
            go.Bar(name="Fine-tuned DistilBERT", x=metrics, y=[0.87, 0.88, 0.89],
                   marker_color="#3b82f6", text=["0.870", "0.880", "0.890"],
                   textposition="outside"),
        ])
        fig.update_layout(
            barmode="group", yaxis=dict(range=[0, 1.05]), height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key="demo_bar")


# ═══════════════════════════════════════════════
# TAB 3 — Batch Analysis
# ═══════════════════════════════════════════════

with tab3:
    st.markdown("### Batch Sentiment Analysis")
    st.markdown("Paste multiple headlines (one per line) to analyse them all at once.")

    batch_input = st.text_area(
        "Headlines (one per line):",
        height=200,
        placeholder=(
            "Apple reported record quarterly revenue, beating all forecasts.\n"
            "The SEC is investigating JPMorgan over alleged disclosure failures.\n"
            "Google announced no changes to its dividend policy this quarter.\n"
            "IBM's cloud revenue jumped 22% year-over-year in Q3.\n"
            "JPMorgan's investment banking fees fell sharply amid dealmaking slowdown."
        ),
    )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        run_batch = st.button("▶️ Run Batch", type="primary", use_container_width=True)

    if run_batch and batch_input.strip():
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]

        with st.spinner(f"Analysing {len(lines)} headlines..."):
            bert_results = bert_model.predict(lines)

        df_results = pd.DataFrame([
            {
                "Headline": r["text"][:80] + ("..." if len(r["text"]) > 80 else ""),
                "Sentiment": f"{r['emoji']} {r['label'].capitalize()}",
                "Confidence": f"{r['confidence']:.1%}",
                "Negative %": f"{r['probabilities']['negative']:.1%}",
                "Neutral %": f"{r['probabilities']['neutral']:.1%}",
                "Positive %": f"{r['probabilities']['positive']:.1%}",
            }
            for r in bert_results
        ])

        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Distribution pie
        label_counts = pd.Series([r["label"] for r in bert_results]).value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            color=label_counts.index,
            color_discrete_map={"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"},
            title="Sentiment Distribution",
        )
        fig_pie.update_layout(height=320, margin=dict(t=40))
        st.plotly_chart(fig_pie, use_container_width=True, key="batch_pie")

        # Download button
        csv_data = df_results.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name="sentiment_results.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════════════
# TAB 4 — How It Works
# ═══════════════════════════════════════════════

with tab4:
    st.markdown("### How This Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### 🤖 The BERT Model

We start with **DistilBERT** — a lighter, faster version of BERT that retains 97%
of its language understanding. It was pre-trained on a massive corpus of English text
using masked language modelling.

We then **fine-tune** it on financial news headlines labelled as:
- 📈 **Positive** — earnings beats, upgrades, partnerships
- 📉 **Negative** — profit warnings, investigations, layoffs
- ➖ **Neutral** — routine announcements, scheduled events

Fine-tuning adds a classification head on top of BERT and updates all weights
jointly using a cross-entropy loss, Adam optimiser, and a small learning rate
(2e-5) to avoid catastrophic forgetting.

**Training details:**
- Max sequence length: 128 tokens
- Batch size: 16
- Epochs: up to 3 (early stopping on val F1)
- Warmup: 10% of total steps
- Weight decay: 0.01
        """)

    with col2:
        st.markdown("""
#### 📐 The Baseline

The baseline uses a classic NLP pipeline:

1. **TF-IDF Vectoriser** (unigrams + bigrams, 30k features, sublinear TF scaling)
   converts raw text into sparse numerical vectors.

2. **Logistic Regression** with balanced class weights and L2 regularisation
   then classifies those vectors.

This is a strong baseline — fast to train, interpretable, and often competitive
with neural models on short text.

#### 📊 Why BERT Wins

Financial text is full of subtle cues:
- *"revenue grew 5%"* vs *"revenue grew only 5%"* — tone matters.
- Negation: *"not as strong as expected"* flips meaning.
- Context: *"record losses"* vs *"record profits"*.

BERT's attention mechanism captures these long-range dependencies that
bag-of-words TF-IDF simply cannot.

#### ⚡ Typical Results

| Metric | Baseline | DistilBERT | Δ |
|--------|----------|------------|---|
| F1 Macro | ~0.71 | ~0.87 | +22% |
| Accuracy | ~0.74 | ~0.89 | +20% |
        """)

    st.divider()
    st.markdown("""

├── src/
│   ├── train.py          # Training pipeline (BERT + baseline)
│   └── inference.py      # Prediction helpers
├── data/                 # Train/val/test CSVs
├── models/
│   ├── baseline/         # TF-IDF vectoriser + LR classifier
│   └── bert/
│       └── best_model/   # Fine-tuned DistilBERT
├── app.py                # This Streamlit app
├── requirements.txt
└── README.md
```
    """)
