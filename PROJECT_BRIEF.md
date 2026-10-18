# GlucoBloom: Personalized Digital Twin for Glucose Forecasting

GlucoBloom is an AI-powered personalized digital twin prototype for continuous glucose forecasting. Rather than functioning simply as a retrospective logging tool, GlucoBloom learns general metabolic dynamics across diverse individuals, then adapts to an individual's unique physiology to forecast glucose levels 30 minutes ahead, provide calibrated uncertainty intervals, explain predictions, and simulate what-if scenarios.

## Repository Structure
```
glucobloom/
├── data/              # CGM datasets & synthetic physiological generators (NEVER commit raw clinical data)
├── model/             # Preprocessing, baseline, XGBoost, quantile regressions, fine-tuning
├── api/               # FastAPI backend implementing the prediction contract
├── website/           # Streamlit interactive web application
├── outputs/           # Serialized metrics, time-machine traces, and twin evolution traces
├── docs/              # Problem statement, dataset documentation, and evaluation results
└── run.py             # One-click launcher for API and Website
```

## Shared Prediction Contract
Every prediction follows the unified schema:
```json
{
  "patient_id": "p8",
  "timestamp": "2026-10-18T12:30:00",
  "predicted_glucose": 142.5,
  "lower": 126.0,
  "upper": 159.0,
  "model": "personalized"
}
```

## Ethics, Privacy & Medical Disclaimer
* **No Medical Advice:** Research and educational prototype only. Not for clinical treatment.
* **Fair Evaluation:** Strictly split by patient to prevent data leakage. Always evaluated against persistence baselines.
* **Privacy:** De-identified data; no personal health identifiers are stored.
