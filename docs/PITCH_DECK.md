# GlucoBloom: Competition Pitch Deck & Demo Script

---

## Slide 1: Title & Hook
* **Title:** GlucoBloom: AI-Powered Personalized Digital Twin for Glucose Forecasting
* **Tagline:** Moving Beyond Retrospective Tracking to Prospective, Personalized Glycemic Intelligence.
* **Team:** Person A (AI & Machine Learning) & Person B (Full-Stack Clinical UI)
* **Hook:** *"Most continuous glucose monitors tell you where you were 5 minutes ago. But diabetes doesn't happen in reverse. What patients and endocrinologists need is to know where glucose is heading 30 minutes in advance, how confident the model is, and how their individual body behaves."*

---

## Slide 2: The Core Problem
* **Limitation of Existing CGM Apps:** Static charts with zero predictive anticipation.
* **The Clinical Bottleneck:** Human metabolism is non-linear. Two people eating the exact same bowl of oatmeal will have drastically different glycemic spikes due to differing insulin sensitivity, gastric emptying, and circadian cortisol rhythms.
* **Why One-Size-Fits-All Models Fail:** Population-level models either under-predict spikes or trigger false hypoglycemia alarms, leading to alarm fatigue and dangerous insulin over-corrections.

---

## Slide 3: The Solution — 2-Level Digital Twin Architecture
1. **Level 1: General Foundation Model (XGBoost Regressor)**
   * Trained on diverse multi-patient cohorts to capture universal glycemic kinetics, diurnal basals, and rate-of-change dynamics.
2. **Level 2: Personalized Digital Twin Adapter**
   * Rapidly fine-tunes on an individual’s own history without catastrophic forgetting.
   * Proves that personalization actually helps through quantitative tracking.
3. **Calibrated Uncertainty (Quantile Regression):**
   * Predicts 10th and 90th percentiles to provide an **80% confidence interval cone** rather than a deceptive single point.

---

## Slide 4: Quantitative Benchmarks & Results (Held-Out Unseen Patients)
*Evaluated on 25 held-out patients (93,568 readings) with zero patient leakage during training:*

| Metric / Model | Baseline (Persistence) | Linear Regression | XGBoost General | Personalized Twin (Patient 175) |
| :--- | :---: | :---: | :---: | :---: |
| **MAE (mg/dL)** | 17.20 | 10.89 | **7.64** | **6.44** |
| **Relative Improvement** | *0.0%* | +36.7% | **+55.6%** | **+26.4% Personalization Gain** |
| **80% Quantile Coverage** | — | — | **85.0%** | **85.0%** |
| **Clarke Error Grid (Zones A+B)** | — | — | **100.0%** | **100.0% (Zero Dangerous Errors)** |

---

## Slide 5: The 5 Signature Features (Live Demo Flow)
1. **🔮 30-Minute Forecast Cone:** Live interactive forecast with shaded 80% prediction interval.
2. **🧬 Digital Twin Evolution:** Visual proof of error dropping from 8.74 down to 6.44 mg/dL as personal data accumulates (Day 1 $\to$ 8).
3. **⏳ Glucose Time Machine & Clarke Grid:** Step-by-step retrospective audit against reality, validated by gold-standard Clarke Error Grid Analysis (100% in Zone A+B).
4. **🧪 Scenario Studio ("Fan of Futures"):** 3 diverging forward trajectories comparing sedentary vs light walk vs brisk exercise after a meal.
5. **🛡️ Smart Data Quality Checker:** Real-time sensor anomaly, compression spike, and missing-reading dropout detector.

---

## Slide 6: Honesty & Ethics Framework
* **Transparent Trust Badges:** Every feature explicitly labelled `[🟢 WORKING]`, `[🟡 DEMONSTRATION]`, or `[⚪ FUTURE]`.
* **Fair Evaluation:** Strictly patient-level holdouts; never random row shuffling.
* **Privacy & Data Sovereignty:** Local processing, `.gitignore` raw data protection, zero PHI retention, and one-click data purge.
* **Medical Disclaimer:** Clear UI banner stating prototype is educational and gives no clinical treatment advice.

---

## 3-Minute Demo Script (Word-for-Word Guide)

### 0:00 - 0:45: The Problem & The Idea
> *"Good morning judges. Managing glucose is like driving a car by only looking in the rear-view mirror. Existing continuous glucose monitors tell you what already happened, but human bodies react differently to meals, exercise, and circadian rhythms.
> 
> We built **GlucoBloom**, an AI-powered personalized digital twin. Instead of just tracking past numbers, it forecasts glucose 30 minutes into the future with calibrated uncertainty, explains its reasoning, and personalizes to each individual."*

### 0:45 - 1:45: Live Dashboard & The Forecast Cone
> *"Let's look at the live dashboard. Here in Tab 1, we see our past 1-hour sensor history. Instead of guessing a single number, GlucoBloom generates an **80% prediction cone** using quantile regression. If our glucose is currently 168 mg/dL and rising fast, the model predicts a peak at 185 mg/dL in 30 minutes, bounded between 165 and 192 mg/dL.
> 
> Furthermore, users can drag and drop their own CGM CSV file right in the sidebar to inspect their own trends immediately."*

### 1:45 - 2:30: Digital Twin Evolution & Clarke Safety Grid
> *"Now for our signature breakthrough: **Digital Twin Personalization** in Tab 2. When an unseen patient first uses GlucoBloom, the general model achieves an MAE of 8.74 mg/dL. As the twin sees 1 day, 3 days, and 8 days of personal data, the error drops down to **6.44 mg/dL — an immediate 26.4% personalization gain**.
> 
> To prove clinical safety, Tab 3 shows the **Clarke Error Grid Analysis**. Across our holdout evaluations, **100% of predictions fall in Zones A and B** (99.6% in Zone A alone), meaning zero clinically dangerous errors."*

### 2:30 - 3:00: Scenario Studio & Conclusion
> *"Finally, in Tab 4, our **Fan of Futures** lets users explore what-if choices: seeing the red sedentary curve spike to 210 mg/dL after pizza, compared to the green curve where a 30-minute walk blunts the spike down to 155 mg/dL.
> 
> Every feature is honestly badged as Working, Demonstration, or Future. GlucoBloom is credible, clinically grounded, and ready to transform glycemic forecasting. Thank you!"*
