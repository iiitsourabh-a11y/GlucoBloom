"""
GlucoBloom Synthetic Physiological CGM Generator
Generates realistic multi-patient continuous glucose monitoring (CGM) time series.
Models:
- Circadian basal dynamics (diurnal drift, dawn phenomenon)
- Carbohydrate intake & two-compartment postprandial glycemic excursions
- Physical activity & acute glucose clearance
- Patient-specific insulin/carbohydrate sensitivity parameters
- Sensor noise & realistic dropout gaps (sensor warmup, compression)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_patient_cgm(patient_id: str, days: int = 21, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Patient physiological profile
    basal_glucose = np.random.uniform(95.0, 135.0)  # mg/dL
    carb_sensitivity = np.random.uniform(1.8, 3.2)  # mg/dL spike per 10g carbs
    clearance_rate = np.random.uniform(0.015, 0.030) # recovery rate per min
    
    start_time = datetime(2026, 9, 1, 0, 0, 0)
    total_steps = days * 288 # 5-min intervals: 288 per day
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(total_steps)]
    
    glucose = np.zeros(total_steps)
    meals = np.zeros(total_steps)
    activity = np.zeros(total_steps)
    
    # Active metabolic pools
    current_glucose = basal_glucose
    gut_carbs = 0.0
    active_exercise = 0.0
    
    for step in range(total_steps):
        t = timestamps[step]
        hour = t.hour + t.minute / 60.0
        
        # Diurnal / Circadian basal oscillation (Dawn phenomenon around 5-8 AM)
        circadian_offset = 8.0 * np.sin(2 * np.pi * (hour - 3) / 24.0) + \
                           (6.0 if 5 <= hour <= 8 else 0.0)
        
        # Random meals
        meal_g = 0.0
        # Breakfast (7:00 - 8:30)
        if t.hour == 7 and t.minute == 30 and np.random.rand() < 0.90:
            meal_g = np.random.uniform(35, 65)
        # Lunch (12:30 - 13:30)
        elif t.hour == 12 and t.minute == 45 and np.random.rand() < 0.90:
            meal_g = np.random.uniform(45, 85)
        # Afternoon snack (16:00)
        elif t.hour == 16 and t.minute == 0 and np.random.rand() < 0.35:
            meal_g = np.random.uniform(15, 30)
        # Dinner (19:30 - 20:30)
        elif t.hour == 19 and t.minute == 30 and np.random.rand() < 0.92:
            meal_g = np.random.uniform(50, 95)
            
        meals[step] = meal_g
        gut_carbs += meal_g
        
        # Activity (walk or gym around 17:30 or 08:30)
        act_mins = 0.0
        if t.hour == 17 and t.minute == 45 and np.random.rand() < 0.40:
            act_mins = 30.0
            active_exercise += act_mins
        activity[step] = act_mins
        
        # Physiological transition over 5 minutes
        # 1. Carbs absorption into blood
        absorbed = gut_carbs * 0.08
        gut_carbs -= absorbed
        spike = absorbed * carb_sensitivity
        
        # 2. Exercise clearance
        exercise_drop = active_exercise * 0.4
        active_exercise *= 0.85
        
        # 3. Endogenous insulin return towards basal
        target_baseline = basal_glucose + circadian_offset
        return_to_baseline = (target_baseline - current_glucose) * clearance_rate * 5.0
        
        # 4. Sensor noise
        noise = np.random.normal(0, 1.2)
        
        current_glucose += spike - exercise_drop + return_to_baseline + noise
        current_glucose = float(np.clip(current_glucose, 45.0, 380.0))
        glucose[step] = round(current_glucose, 1)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": timestamps,
        "glucose": glucose,
        "meal_carbs": meals,
        "activity_mins": activity
    })
    
    # Introduce realistic sensor dropouts (randomly drop ~1.5% of samples)
    drop_mask = np.random.rand(len(df)) < 0.015
    # Also introduce one 2-hour sensor warmup gap on day 10
    warmup_start = 10 * 288 + 40
    drop_mask[warmup_start:warmup_start + 24] = True
    
    df_with_gaps = df[~drop_mask].copy().reset_index(drop=True)
    return df_with_gaps

def main():
    print("Generating multi-patient CGM dataset (10 patients, 21 days each)...")
    patients_data = []
    for i in range(1, 11):
        pid = f"p{i}"
        pdf = generate_patient_cgm(pid, days=21, seed=100 + i)
        patients_data.append(pdf)
        print(f"  Generated {pid}: {len(pdf)} valid readings")
        
    full_df = pd.concat(patients_data, ignore_index=True)
    out_path = "glucobloom/data/cgm_multi_patient.csv"
    full_df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path} with {len(full_df)} total records across {full_df['patient_id'].nunique()} patients.")

if __name__ == "__main__":
    main()
