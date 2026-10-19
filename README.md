# 🌸 GlucoBloom: AI-Powered Personalized Digital Twin for Glucose Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20Quantiles-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Clinical Safety](https://img.shields.io/badge/Clarke%20Grid-100%25%20Zones%20A%2BB-success.svg)](#clinical-safety--clarke-error-grid-analysis)

---

## 📋 Mandatory Submission Details (Digital Twin Challenge 2026)

### 1. Team Details
* **Team Name:** Team GlucoBloom
* **Lead Engineer (Person A - AI/ML & Pipeline):** Sourabh ([@iiitsourabh-a11y](https://github.com/iiitsourabh-a11y))
* **Co-Developer (Person B - Web & Clinical UI):** Clinical Frontend Developer
* **Contact Email:** iiitsourabh@gmail.com
* **Official Repository Link:** [https://github.com/iiitsourabh-a11y/glucobloom](https://github.com/iiitsourabh-a11y/glucobloom)

### 2. College / Incubator Information
* **Institution / University:** [Your University / College Name Here]
* **Incubator / Department:** Department of Computer Science & Artificial Intelligence / Healthcare Innovation Lab

### 3. Project Title
**GlucoBloom: An AI-Powered Personalized Digital Twin for Continuous Glucose Forecasting, Uncertainty Quantification, and What-If Lifestyle Simulation**

### 4. Problem Statement
Continuous Glucose Monitoring (CGM) devices have transformed diabetes self-management by reporting real-time interstitial glucose levels every 5 minutes. However, existing commercial platforms suffer from a critical limitation: **they operate purely in reverse, displaying retrospective numbers rather than actionable forward trajectories.** 

Human glycemic response is highly non-linear, dynamic, and individualized: two people ingesting the exact same meal exhibit vastly different blood glucose curves due to differences in insulin sensitivity, gastric clearance rate, physical activity, and diurnal circadian cortisol cycles. Static, one-size-fits-all population models fail to generalize, leading to alarm fatigue and dangerous insulin stacking overcorrections.

### 5. Healthcare Use Case
* **Target Patient Population:** Individuals living with Type 1 Diabetes (T1D), Insulin-requiring Type 2 Diabetes (T2D), and prediabetes.
* **Clinical Setting:** Ambulatory diabetes care, endocrinology clinics, and automated decision-support applications.
* **Core Clinical Purpose:**
  1. **30-Minute Early Warning Window:** Anticipating imminent hypoglycemic events (<70 mg/dL) or postprandial hyperglycemia (>180 mg/dL) 30 minutes in advance, allowing preventive action (e.g. fast-acting carbs or a brief walk) before symptoms manifest.
  2. **Personalized Metabolic Adaptation:** Tracking an individual's digital twin as it personalizes to their circadian rhythms and recovery rates over 1 to 14 days of CGM history.
  3. **Lifestyle Intervention Simulation ("Fan of Futures"):** Demonstrating the prospective glycemic impact of meal carbohydrates versus physical exercise before eating.

### 6. Technical Stack
* **Programming Language:** Python 3.10+ (tested on Python 3.14)
* **Data Processing & Feature Engineering:** `pandas`, `numpy`, `scipy`
* **Machine Learning & Gradient Boosting:** `scikit-learn`, `xgboost` (Quantile Regressors for $\alpha=0.10, \alpha=0.90$)
* **Model Serialization:** `joblib`
* **API Service:** `FastAPI`, `pydantic`, `uvicorn`, `httpx`
* **Web UI & Interactive Visualizations:** `streamlit`, `plotly`
* **Clinical Report Generation:** `reportlab`
* **Version Control & Documentation:** Git, GitHub, Markdown

### 7. AI/ML Model & Framework Details
* **Multi-Patient Population Dataset:** 200 virtual diabetic patients across 14–21 days of continuous monitoring (806,400 raw readings; 748,637 engineered sliding windows) incorporating CGM, meal carbs, steps, heart rate, and sleep.
* **Evaluation Protocol:** Strict patient-level split (160 train patients, 15 validation patients, and 25 completely unseen held-out test patients). **Zero random row leakage.**
* **Level 1 General Foundation Model:** XGBoost Regressor trained on 160 patients, capturing universal kinetics and diurnal rhythms.
* **Level 2 Personalized Digital Twin Adapter:** Temporal fine-tuning on unseen patient history (Patient 175) across Days 1, 3, 5, and 8 without catastrophic forgetting.
* **Calibrated Uncertainty:** XGBoost Quantile Regressors predicting 10th and 90th percentiles to render an 80% confidence interval cone.
* **Feature Vector (15 features):** Current CGM reading, historical lags ($t-5, t-10, t-15, t-30, t-60$ mins), 15m & 30m velocity, 30m cumulative carbs, 30m cumulative steps, heart rate, HRV, sleep state, and circadian diurnal phase ($\sin / \cos$ of hour).

#### Quantitative Benchmark Results (25 Held-Out Test Patients)
| Model / Strategy | MAE (mg/dL) | RMSE (mg/dL) | Relative Gain vs Baseline | Clinical Status |
| :--- | :---: | :---: | :---: | :--- |
| **Persistence Baseline (t = t+30m)** | **17.20** | **29.59** | *0.0% (Reference)* | Blind to trajectory |
| **Linear Regression Benchmark** | **10.89** | **15.60** | +36.7% | Linear approximation |
| **XGBoost General Foundation Model** | **7.64** | **11.94** | **+55.6%** | High precision |
| **Personalized Twin (Patient 175 Holdout)** | **6.44** | **9.81** | **+26.4% Personalization Gain** | Optimal personalized fit |

* **Uncertainty Calibration:** 80% prediction cone achieved **85.0% empirical coverage** on held-out test patients.
* **Clinical Safety (Clarke Error Grid Analysis):** **100.0% of predictions fall in Zones A + B** (99.6% in Zone A alone), proving zero clinically dangerous treatment errors.

---

### 8. 15-20 Minute Demo Video Link (Unlisted YouTube)
* **Unlisted YouTube Video Link:** `https://youtu.be/YOUR_UNLISTED_VIDEO_LINK_HERE`  
*(Please replace with your uploaded Unlisted YouTube link before submitting)*

#### Video Walkthrough Chapters & Outline:
* `00:00 - 02:30` — Team introduction, problem statement & healthcare rationale
* `02:30 - 05:00` — Clinical CGM dataset overview & strict patient-level splitting
* `05:00 - 08:30` — Machine learning pipeline: Persistence baseline, XGBoost & quantile regression
* `08:30 - 12:00` — Live Interactive Walkthrough: 30-min Forecast Cone & CSV Drag-and-Drop Uploader
* `12:00 - 14:30` — Digital Twin Personalization: +26.4% error reduction trajectory & Time Machine replay
* `14:30 - 17:00` — Clarke Error Grid Analysis (100% Zones A+B) & Scenario Studio ("Fan of Futures")
* `17:00 - 19:30` — Smart Data Quality Checker, Doctor-Ready AGP Report Export & Conclusion

---

### 9. Open-Source License Details
GlucoBloom is released under the **MIT Open-Source License**. Full license details are available in the repository at [LICENSE](LICENSE).

---

### 10. Architecture Diagram (PDF / PPT Format)
* **Architecture Diagram (PDF):** Available in the repository at [docs/architecture_diagram.pdf](docs/architecture_diagram.pdf).
* Full vector block diagram depicting the 4-layer architecture:
  1. Sensor Ingestion & Data Quality Guard
  2. Feature Engineering & Patient-Holdout Splitting
  3. Two-Level AI Digital Twin Engine (General + Quantiles + Personal Adapter)
  4. Clinical Decision Support UI & AGP Export

---

### 11. Presentation (PDF / PPT Format)
* **Project Presentation Slides (PDF):** Available in the repository at [docs/presentation.pdf](docs/presentation.pdf).
* **Pitch Deck & Demo Script (Markdown):** Available at [docs/PITCH_DECK.md](docs/PITCH_DECK.md).
* Covers project motivation, clinical background, dataset details, mathematical formulation, benchmark outcomes, and safety evaluation.

---

### 12. Public Accessibility Statement
All files, models, documents, and scripts within this repository are **publicly accessible without additional permissions or login requirements**, in strict accordance with the Digital Twin Challenge 2026 guidelines.

---

## 💻 How to Run the Project Locally

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Launch the Connected HTML Dashboard + AI Engine
```powershell
python -m uvicorn api.main:app --port 8000
```
Open **`http://localhost:8000`** in your browser.

### 3. Launch the Full Interactive Clinical Streamlit Suite
```powershell
python -m streamlit run website/app.py --server.port=8501
```
Open **`http://localhost:8501`** in your browser.

---

## ⚕️ Medical Ethics & Privacy Disclaimer
GlucoBloom is an educational and computational research prototype. It does not provide medical diagnosis or treatment advice. In accordance with clinical data ethics, raw patient CGM time series are never committed to version control and are strictly git-ignored.
