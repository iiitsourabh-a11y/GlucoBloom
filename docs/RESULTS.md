# Experimental Results & Model Evaluation Benchmark

## 1. Metric Comparison on Unseen Patients (p8, p9, p10)
Evaluation protocol: Strict patient-level holdout. 30-minute forward forecast ($t+6$ samples).

| Model / Strategy | MAE (mg/dL) | RMSE (mg/dL) | Relative Improvement vs. Baseline |
| :--- | :---: | :---: | :---: |
| **Persistence Baseline (Future = Current)** | **6.99** | **11.36** | *0.0% (Reference)* |
| **Linear Regression Benchmark** | **5.22** | **8.93** | +25.3% |
| **XGBoost General Foundation Model** | **3.23** | **4.97** | **+53.7%** |

## 2. Uncertainty Quantification (Quantile Regression)
* **Objective:** Calibrated 80% Prediction Interval ($10^{\text{th}}$ to $90^{\text{th}}$ percentile bounds).
* **Empirical Coverage on Unseen Holdout:** **94.0%** of actual glucose readings fell within the forecasted cone.

## 3. Digital Twin Personalization Trajectory (Patient p8)
Evaluated on unseen holdout days (Days 15–21) after temporal fine-tuning:

| Personal Data Seen | Samples Accumulated | Personal Twin MAE (mg/dL) | General Model MAE (mg/dL) | Personalization Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Day 0 (General Baseline)** | 0 | 3.14 | 3.14 | 0.0% |
| **Day 1** | 288 | 3.05 | 3.14 | +3.0% |
| **Day 3** | 864 | 3.03 | 3.14 | +3.7% |
| **Day 7** | 2,016 | **2.87** | 3.14 | **+8.8%** |
| **Day 14** | 4,032 | 2.90 | 3.14 | +7.8% |

*Key Finding:* Personalization yields peak performance improvement (+8.8%) once ~1 week of circadian rhythms and individual carb-clearance responses are integrated into the twin.
