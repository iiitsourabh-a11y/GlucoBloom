# Experimental Results & Model Evaluation Benchmark
*Trained and Evaluated on 200 Virtual Diabetic Patients (748,637 engineered samples)*

## 1. Metric Comparison on 25 Unseen Test Patients (Patients 175 – 199)
Evaluation protocol: Strict patient-level holdout. 30-minute forward forecast ($t+6$ samples at 5-minute sampling).

| Model / Strategy | MAE (mg/dL) | RMSE (mg/dL) | Relative Improvement vs. Baseline |
| :--- | :---: | :---: | :---: |
| **Persistence Baseline (Future = Current)** | **17.20** | **29.59** | *0.0% (Reference)* |
| **Linear Regression Benchmark** | **10.89** | **15.60** | +36.7% |
| **XGBoost General Foundation Model** | **7.64** | **11.94** | **+55.6%** |

## 2. Uncertainty Quantification (Quantile Regression)
* **Objective:** Calibrated 80% Prediction Interval ($10^{\text{th}}$ and $90^{\text{th}}$ percentile bounds).
* **Empirical Coverage on 25 Unseen Test Patients:** **85.0%** of actual future readings fall within the predicted cone.

## 3. Digital Twin Personalization Trajectory (Patient 175)
Evaluated on unseen holdout days (Days 10–13) after temporal adaptation:

| Personal Data Seen | Samples Accumulated | Personal Twin MAE (mg/dL) | General Model MAE (mg/dL) | Personalization Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Day 0 (General Baseline)** | 0 | 8.74 | 8.74 | 0.0% |
| **Day 1** | 262 | 6.94 | 8.74 | +20.6% |
| **Day 3** | 787 | 6.49 | 8.74 | +25.7% |
| **Day 5** | 1,335 | 6.53 | 8.74 | +25.3% |
| **Day 8** | 2,136 | **6.44** | 8.74 | **+26.4%** |

*Key Finding:* Adapting the general gradient booster to an individual's personal glucose history slashes prediction error by **26.4%** (from 8.74 down to 6.44 mg/dL).
