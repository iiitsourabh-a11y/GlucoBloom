# Problem Statement (PS): GlucoBloom

## 1. The Challenge
Continuous Glucose Monitoring (CGM) devices provide real-time interstitial glucose readings every 5 minutes. However, existing consumer tools primarily show retrospective data ("what happened"), leaving patients and clinicians guessing about near-term trajectories. 

Glycemic responses are highly non-linear and individualized: identical meals cause widely varying spikes across people due to differing insulin sensitivity, circadian rhythms, and metabolic lag. Static, one-size-fits-all predictive models fail to generalize and can foster misplaced trust.

## 2. The Solution: Personalized Digital Twin
GlucoBloom introduces a 2-level predictive architecture:
1. **General Foundation Model:** Trained on diverse multi-patient cohorts to capture universal glycemic kinetics, diurnal cycles, and recovery rates.
2. **Personalized Digital Twin Adapter:** Rapidly personalizes to an individual's glycemic response curve using temporal fine-tuning without catastrophic forgetting.

## 3. Core Capabilities
* **30-Minute Forward Prediction Cone:** Forecasts glucose 30 minutes ahead ($t+6$ readings) with an 80% confidence interval via calibrated quantile regression.
* **Digital Twin Evolution:** Demonstrates progressive accuracy gains as days of personal CGM history accumulate.
* **Glucose Time Machine:** Transparently replays historic forecasts against actual measurements to verify reliability.
* **Explainable AI (XAI):** Deconstructs forecasts into contributing features (velocity, rolling trend, circadian phase).
* **Scenario Studio:** Explores what-if lifestyle adjustments (e.g., high vs. low carb impact, post-meal light physical activity).

## 4. Honesty Principle
Every feature in GlucoBloom is strictly categorized:
* `[WORKING]`: Tested against real/simulated unseen patient holdouts with true quantitative metrics.
* `[DEMONSTRATION]`: Simulated illustrative workflows where full clinical sensor logs are unavailable.
* `[FUTURE]`: Planned roadmap features (e.g. real-time automated insulin pump telemetry).
