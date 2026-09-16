# 🛡️ Network Intrusion Detection System (NIDS)

A machine learning based Network Intrusion Detection System that classifies network traffic as **normal** or **malicious**, detects attack categories, and exposes predictions through a **FastAPI** backend with an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

Traditional signature-based intrusion detection systems fail against novel attacks. This project applies **supervised machine learning** to classify network traffic flows using the **NSL-KDD** dataset, one of the most widely used benchmarks in intrusion detection research.

The system:

1. Loads and preprocesses the NSL-KDD dataset
2. Trains multiple ML models (Random Forest, SVM, Logistic Regression)
3. Evaluates models on accuracy, precision, recall, and F1
4. Classifies traffic as **Normal** or **Attack** (binary)
5. Predicts the **attack category** (DoS, Probe, R2L, U2R) in multi-class mode
6. Exposes predictions through a FastAPI REST API
7. Visualizes attack patterns and model metrics in an interactive dashboard

---

## Features

### Data Processing
- Handles NSL-KDD categorical features (protocol_type, service, flag)
- Feature scaling and encoding pipeline
- Train/test split with stratification
- Class imbalance analysis and reporting

### Machine Learning
- **Random Forest** classifier (primary model)
- **Logistic Regression** and **SVM** for comparison
- Binary classification: Normal vs Attack
- Multi-class classification: DoS, Probe, R2L, U2R, Normal
- Model evaluation with confusion matrix, precision, recall, F1

### Analysis & Visualization
- Attack distribution charts
- Feature importance ranking
- Confusion matrix heatmaps
- Model comparison table
- Per-class performance metrics

### Deployment
- FastAPI backend with prediction endpoint
- Interactive dashboard for live prediction
- Saved model artifacts (`.pkl`) for inference

---

## Dataset

**NSL-KDD** — an improved version of the KDD Cup 1999 dataset, commonly used in IDS research.

- **Source:** https://www.unb.ca/cic/datasets/nsl.html
- **Training set:** `KDDTrain+.txt` (~125,000 records)
- **Test set:** `KDDTest+.txt` (~22,500 records)
- **Features:** 41 (connection duration, protocol, service, flags, byte counts, etc.)
- **Labels:** Normal or one of 4 attack categories:
  - **DoS** — Denial of Service
  - **Probe** — Surveillance / scanning
  - **R2L** — Remote to Local
  - **U2R** — User to Root

### Download and placement

```
data/
├── raw/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
└── processed/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    └── y_test.csv
```

Download the raw files manually from the UNB link above and place them in `data/raw/`.

---

## Architecture

```
┌────────────────────────┐
│  NSL-KDD Dataset       │
│  (KDDTrain+, KDDTest+) │
└───────────┬────────────┘
            │
     ┌──────▼──────┐
     │ Preprocess  │   encode categoricals · scale numerics
     └──────┬──────┘
            │
     ┌──────▼──────┐
     │ Train/Test  │   stratified split
     │   Split     │
     └──────┬──────┘
            │
   ┌────────┼────────┐
   │        │        │
┌──▼───┐ ┌──▼───┐ ┌──▼────┐
│  RF  │ │ SVM  │ │ LogReg│
└──┬───┘ └──┬───┘ └──┬────┘
   │        │        │
   └────────┼────────┘
            │
     ┌──────▼──────┐
     │  Evaluate   │   accuracy · F1 · confusion matrix
     └──────┬──────┘
            │
     ┌──────▼──────┐
     │ Save Model  │   models/*.pkl
     └──────┬──────┘
            │
   ┌────────┴────────┐
   │                 │
┌──▼────┐      ┌─────▼─────┐
│FastAPI│      │ Dashboard │
│  API  │      │ (predict) │
└───────┘      └───────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | scikit-learn, pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Model storage | joblib / pickle |
| Testing | pytest |

---

## Project Structure

```
NIDS-project/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── predict.py            # inference helpers
│   └── main.py               # entry point
├── api/
│   └── server.py             # FastAPI app
├── src/
│   ├── preprocess.py         # data loading and cleaning
│   ├── train.py              # model training
│   ├── evaluate.py           # metrics and plots
│   └── features.py           # feature engineering
├── notebooks/
│   └── exploratory.ipynb     # EDA and experiments
├── models/
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── screenshots/
├── tests/
│   └── test_pipeline.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11 or newer
- Git
- (Optional) Docker

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NIDS.git
cd NIDS
```

### 2. Create a virtual environment

**Windows:**
```powershell
python -m venv nids_env
nids_env\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv nids_env
source nids_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `KDDTrain+.txt` and `KDDTest+.txt` from https://www.unb.ca/cic/datasets/nsl.html and place them in `data/raw/`.

---

## Usage

### Step 1 — Preprocess the data

```bash
python -m src.preprocess
```

This produces `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

### Step 2 — Train the models

```bash
python -m src.train
```

Expected output:

```
Loading data...
Training Random Forest...
  Accuracy:  0.994
  Precision: 0.993
  Recall:    0.994
  F1:        0.993

Training Logistic Regression...
  Accuracy:  0.923
  ...

Best model: Random Forest — saved to models/random_forest.pkl
```

### Step 3 — Evaluate

```bash
python -m src.evaluate
```

Generates confusion matrices, feature importance plots, and a model comparison table into `docs/screenshots/`.

### Step 4 — Start the API

```bash
uvicorn api.server:app --reload
```

API: http://127.0.0.1:8000/docs

### Step 5 — Start the dashboard

**New terminal:**
```bash
streamlit run app/main.py
```

Dashboard: http://localhost:8501

---

## Model Performance

Evaluated on the NSL-KDD test set.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | ~0.99 | ~0.99 | ~0.99 | ~0.99 |
| Logistic Regression | ~0.92 | ~0.91 | ~0.92 | ~0.91 |
| SVM | ~0.96 | ~0.95 | ~0.96 | ~0.95 |

> **Note:** High accuracy on NSL-KDD is well-documented in the literature — the dataset contains many easily separable attack signatures. Real-world IDS performance is typically lower.

### Confusion Matrix (Random Forest)

![Confusion Matrix](docs/screenshots/confusion_matrix.png)

### Feature Importance

![Feature Importance](docs/screenshots/feature_importance.png)

### Attack Distribution

![Attack Distribution](docs/screenshots/attack_distribution.png)

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/model/info` | Model metadata and metrics |
| POST | `/predict` | Predict a single network flow |
| POST | `/predict/batch` | Predict multiple flows |

### Example — single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450,
    "count": 1,
    "srv_count": 1,
    "serror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0
  }'
```

Response:

```json
{
  "prediction": "normal",
  "attack_category": null,
  "confidence": 0.997
}
```

### Example — attack prediction

```json
{
  "prediction": "attack",
  "attack_category": "DoS",
  "confidence": 0.982
}
```

---

## Dashboard

The Streamlit dashboard provides:

- **Overview:** model metrics, dataset statistics
- **Live Prediction:** enter a network flow manually and get instant classification
- **Attack Analysis:** distribution of attack types in the dataset
- **Model Comparison:** side-by-side accuracy, F1, precision, recall
- **Feature Importance:** top 20 features driving predictions
- **Confusion Matrix:** heatmap per class

---

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Preprocessing pipeline shape and dtypes
- Model loading and prediction
- API endpoint responses
- Edge case handling (missing values, unknown categories)

---

## Limitations

This project is a **research prototype**, not a production IDS.

- **Dataset bias:** NSL-KDD is a 1999-era dataset. Modern attack patterns (ransomware, lateral movement, encrypted C2) are not represented.
- **Offline classification:** The model classifies pre-extracted flow features — it does not perform live packet capture.
- **No real-time pipeline:** Integration with tools like `tcpdump`, `Zeek`, or `Suricata` for live traffic is out of scope.
- **Class imbalance:** U2R and R2L attack categories are severely underrepresented in the training data.
- **Model size:** Random Forest `.pkl` is ~10 MB, not suitable for embedded deployment.
- **No adversarial robustness testing:** Model has not been evaluated against adversarial feature manipulation.

---

## Future Work

- **Real-time packet capture** with `Scapy` or `Zeek`
- **CICIDS2017 / CICIDS2019** dataset for modern traffic
- **Deep learning models** (LSTM, CNN) for sequence-based detection
- **Anomaly detection** (autoencoders) to catch zero-day attacks
- **Docker deployment** with auto-scaling
- **SIEM integration** (Splunk, ELK)
- **Adversarial robustness** evaluation
- **Model explainability** with SHAP

---

## License

Educational / academic use.

---

## Acknowledgements

- **NSL-KDD dataset:** Canadian Institute for Cybersecurity, University of New Brunswick
- **KDD Cup 1999:** Original dataset authors
- **scikit-learn** community for the ML tooling 
