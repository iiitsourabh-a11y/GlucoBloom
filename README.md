# 🌸 GlucoBloom: AI-Powered Personalized Digital Twin for Glucose Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20Quantiles-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%201.66-FF4B4B.svg)](https://streamlit.io/)
[![Clinical Safety](https://img.shields.io/badge/Clarke%20Grid-100%25%20Zone%20A%2BB-success.svg)](#clinical-safety--clarke-error-grid)

> **GlucoBloom** is a clinical AI prototype that personalizes continuous glucose forecasting. Moving beyond retrospective logging, GlucoBloom captures multi-person metabolic dynamics, rapidly adapts to an individual's personal physiology, predicts glucose 30 minutes ahead with calibrated 80% prediction intervals, validates clinical safety via the Clarke Error Grid, and simulates what-if lifestyle scenarios.

---

## 🏆 Key Benchmark Highlights (Evaluated on 25 Unseen Test Patients)

* **Baseline Beating:** Cuts 30-minute forward error from **17.20 mg/dL** (Persistence Baseline) down to **7.64 mg/dL** (XGBoost General Model) — a **55.6% relative improvement**.
* **Personalization Gain:** Fine-tuning on an individual's history yields a **+26.4% accuracy boost** (reducing holdout error from 8.74 down to **6.44 mg/dL**).
* **Uncertainty Calibration:** 80% prediction intervals achieve **85.0% empirical coverage** on held-out patients.
* **Clinical Safety Validation:** **100.0% of predictions fall in Clarke Error Grid Zones A + B** (99.6% in Zone A alone), confirming zero dangerous treatment errors.

---

## 🚀 Quickstart: Running GlucoBloom

### 1. Installation
Clone the repository and install dependencies:
```powershell
cd "D:\Amazon ML Challange 2026\glucobloom"
python -m pip install -r requirements.txt
```
*(Dependencies: `streamlit`, `fastapi`, `uvicorn`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `plotly`, `httpx`)*

### 2. Launch Interactive Web Dashboard
```powershell
python run.py
```
Open your browser to: **`http://localhost:8501`**

### 3. Launch FastAPI Prediction Service (Optional)
```powershell
python run.py api
```
Access interactive Swagger API documentation at: **`http://127.0.0.1:8000/docs`**

---

## 🧭 Repository Architecture

```
glucobloom/
├── data/
│   ├── sample_template.csv         # Ready-to-use CSV template for custom uploads
│   ├── generate_cgm_data.py        # Physiological multi-patient CGM generator
│   └── timeseries.csv              # 200 virtual diabetic patients (806,400 readings, git-ignored)
├── model/
│   ├── train_pipeline.py           # Preprocessing, patient splitting, XGBoost, quantiles & personalization
│   ├── clarke_error_grid.py        # Clarke Error Grid Analysis (EGA) computation
│   ├── glucobloom_general.pkl      # General Foundation Model
│   ├── glucobloom_lo.pkl           # 10th percentile quantile regressor
│   ├── glucobloom_hi.pkl           # 90th percentile quantile regressor
│   └── glucobloom_personalized_p8.pkl # Fine-tuned personalized digital twin
├── api/
│   ├── main.py                     # FastAPI REST API implementing the Shared Contract
│   └── test_api.py                 # Automated unit tests for all endpoints
├── website/
│   └── app.py                      # Full Streamlit 7-module clinical dashboard
├── outputs/                        # Serialized benchmark metrics, time-machine traces, and EGA data
├── docs/
│   ├── PS.md                       # Formal Problem Statement
│   ├── DATA.md                     # Data schema & gap handling specifications
│   ├── RESULTS.md                  # Comprehensive empirical evaluation results
│   └── PITCH_DECK.md               # 3-minute pitch deck & word-for-word demo script
├── run.py                          # Unified single launcher
└── .gitignore                      # Privacy guardrail: blocks raw clinical data from version control
```

---

## 🌟 The 5 Signature Features

1. **🔮 30-Minute Forward Forecast Cone `[🟢 WORKING]`**
   * Projects glucose 30 minutes ahead ($t+6$ readings) with an 80% shaded prediction interval.
   * Includes live CSV upload support for personal CGM datasets.
2. **🧬 Digital Twin Evolution `[🟢 WORKING]`**
   * Shows error decreasing across Day 1, Day 3, Day 5, and Day 8 as the twin learns personal metabolic kinetics.
3. **⏳ Glucose Time Machine & Clarke Error Grid `[🟢 WORKING]`**
   * Scrub through history to audit predictions against actual values.
   * Visualizes clinical safety zones (Zones A through E).
4. **🧪 Scenario Studio ("Fan of Futures") `[🟡 DEMONSTRATION]`**
   * 3 branching pathways: Sedentary (+0 min walk) vs Light Walk (+15 min) vs Active Walk (+30 min).
   * Interactive carbohydrate meal diary.
5. **🛡️ Smart Data Quality Checker `[🟢 WORKING]`**
   * Automatically detects sensor dropouts (>15 min), missing timestamps, and compression artifacts.

---

## ⚕️ Ethics & Medical Disclaimer
GlucoBloom is an educational and research prototype. It does not provide medical advice and must never be used to make clinical diagnosis or treatment decisions.
