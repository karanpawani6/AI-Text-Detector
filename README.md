# AI-Text-Detector

A comparative AI-generated text detection system built on two independent classification pipelines — a from-scratch **Decision Tree** using stylometric features, and a **BERT-MLP** pipeline using deep semantic embeddings. Both models were trained on the MAGE benchmark (170,000 samples across 7 writing domains and ~29 generative models) and are served through a single unified Streamlit demo.

🔗 **Live Demo:** [Click here to try it](https://ai-text-detector-d4dsw5r8fewvevbwuuapphf.streamlit.app/)

---

## Overview

Distinguishing AI-generated text from human writing is increasingly important for academic integrity, misinformation detection, and content authenticity. This project implements and compares two methodologically distinct approaches:

- **Decision Tree (Stylometric):** A from-scratch CART implementation operating on 16 hand-engineered linguistic features (hapax ratio, sentence burstiness, passive voice ratio, etc.). Achieves **79.01% accuracy** and **0.8882 ROC-AUC**, with strong interpretability — every prediction can be traced through an explicit decision path.
- **BERT-MLP (Semantic):** Text is embedded via `bert-base-uncased`, compressed to 80 dimensions via PCA, and classified by a 4-layer MLP (256→128→64→2). Achieves **91.78% accuracy** and **0.9751 ROC-AUC**, capturing deeper semantic and discourse-level signals that stylometry misses.

Both models use a conservative classification threshold (0.70 for the DT) to minimize false positives — since incorrectly flagging genuine human writing as AI-generated carries a much higher cost in real academic settings than missing a case of AI use.

## Demo

The unified `app.py` launches a single Streamlit app with a top switcher to flip between the two models — each with its own full interactive GUI (metrics, feature importance charts, decision path visualization, embedding distributions, etc.).

## Project Structure

```
AI-Text-Detector/
├── app.py                      # Unified entry point — switches between both model GUIs
├── decision_tree/
│   ├── dt_gui.py                # Decision Tree Streamlit interface
│   ├── scripts/
│   │   ├── extract_dt.py        # Stylometric feature extraction
│   │   └── train_dt.py          # Model training script
│   ├── model/
│   │   └── dt_model.pkl         # Trained Decision Tree + scaler + label map
│   └── results/                 # ROC curve, feature importance plots, raw results
├── bert_mlp/
│   ├── ann_gui.py                # BERT-MLP Streamlit interface
│   ├── scripts/
│   │   ├── bert_extract.py       # BERT embedding extraction
│   │   ├── train_mlp.py          # MLP training script
│   │   └── infer_mlp.py          # Inference pipeline
│   ├── model/
│   │   └── *.pkl                 # Trained MLP, PCA, scaler, label encoder
│   └── results/                  # Training stats
├── data/                         # Extracted feature datasets (CSV)
├── reports/
│   └── AI_Text_Detection_Final_Report.docx
├── requirements.txt
└── .gitignore
```

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/karanpawani6/AI-Text-Detector.git
cd AI-Text-Detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the unified app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Use the top switcher to flip between the **Decision Tree** and **BERT-MLP** tabs.

> Note: the first time you use the BERT-MLP tab, it will download the `bert-base-uncased` model (~440MB) — this only happens once.

**Running a single model standalone** (optional):
```bash
streamlit run decision_tree/dt_gui.py
streamlit run bert_mlp/ann_gui.py
```

## Methodology Summary

Both models were trained through three iterative dataset phases, each correcting a flaw discovered in the previous one:

| Phase | Dataset | Issue Found |
|---|---|---|
| 1 | HC3 Reddit QA (18k) | Length confound — AI answers were systematically longer, inflating accuracy to 96–99% |
| 2 | HC3 + Persuasive Essays (34k) | Severe domain overfitting to essay-style writing |
| 3 | MAGE (170k, 7 domains, ~29 models) | Realistic, generalizable evaluation — final configuration |

Full methodology, parameter sweeps, cross-validation results, and failure-case analysis are documented in [`reports/AI_Text_Detection_Final_Report.docx`](reports/AI_Text_Detection_Final_Report.docx).

## Results Summary

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Decision Tree | 79.01% | 0.7760 | 0.8882 |
| BERT-MLP | 91.78% | 0.9178 | 0.9751 |

The BERT-MLP outperforms the Decision Tree across all metrics, but the Decision Tree offers full interpretability via traceable decision paths — useful in contexts where an explanation is required alongside a prediction.

## Team

- **Muhammad Mustafa** (31169)
- **Arham Awan** (30934)
- **Muhammad Ismail** (30917)
- **Karan Kumar** (30212)

Institute of Business Administration, Karachi — *Introduction to Artificial Intelligence*, Dr. Syed Ali Raza, Spring 2026

## License

This project was developed for academic coursework. Feel free to reference the methodology with attribution.
