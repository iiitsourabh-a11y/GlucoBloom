# Dataset Documentation & Preprocessing Specifications

## 1. Data Schema
GlucoBloom operates on multi-patient time-series continuous glucose monitor (CGM) streams:
* `patient_id`: Unique identifier (e.g., `p1`, `p2`, ..., `p10`).
* `timestamp`: ISO-8601 UTC timestamp sampled at ~5-minute intervals.
* `glucose`: Interstitial glucose measurement in `mg/dL` (physiological range: 40–400 mg/dL).
* `meal_carbs`: Carbohydrate intake in grams (optional / lifestyle log).
* `activity_mins`: Moderate-to-vigorous exercise in minutes (optional).

## 2. Preprocessing & Gap Handling
1. **Regularization:** Resample raw timestamps onto a strict 5-minute grid.
2. **Short Gap Imputation:** Gaps $\le 15$ minutes (up to 3 consecutive missed samples) are linearly interpolated.
3. **Long Gap Masking:** Gaps $> 15$ minutes are flagged; any 12-reading feature window or target reading overlapping an unfillable gap is discarded to preserve temporal integrity.
4. **Sensor Noise & Outlier Clipping:** Out-of-bounds readings (< 40 or > 450 mg/dL) are flagged for sensor error.

## 3. Train / Validation / Test Splitting Protocol
* **Strict Patient-Level Split:** Models are trained on Patients `p1` through `p7`, validated on `p7`, and evaluated on unseen held-out Patients `p8`, `p9`, `p10`.
* **Random Row Splitting is Prohibited:** Random splits leak future patient temporal correlations into training, artificially inflating accuracy.
* **Temporal Holdout for Personalization:** For test patients, the first $N$ days are allocated for digital twin adaptation, and the remaining holdout days are reserved strictly for evaluation.
