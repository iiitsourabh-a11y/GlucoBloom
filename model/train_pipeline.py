"""
GlucoBloom Training & Digital Twin Personalization Pipeline
Follows strict evaluation protocol:
- Regularizes 5-min intervals and masks gaps >15 mins
- Builds 12-reading sliding windows (1 hr) with velocity & circadian features
- Target: 30 minutes ahead (6 steps)
- Evaluates Persistence Baseline, Linear Regression, and XGBoost
- Trains Quantile Regressors (alpha=0.10, alpha=0.90) for 80% confidence interval
- Strict patient-level split (p1-p6 train, p7 val, p8-p10 unseen test)
- Implements Digital Twin Personalization on unseen patient (Day 1, 3, 7, 14 adaptation)
- Exports models and time-machine JSON payloads
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

FEATURE_NAMES = [f"g_{i}" for i in range(12)] + [
    "velocity_15m", "velocity_30m", "mean_1hr", "std_1hr", "min_1hr", "max_1hr", "hour_sin", "hour_cos"
]

def preprocess_and_window(df: pd.DataFrame, past_steps: int = 12, ahead_steps: int = 6):
    """Resamples patient series, handles gaps, and constructs engineered feature windows."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    patient_dfs = []
    
    for pid, group in df.groupby("patient_id"):
        g = group.sort_values("timestamp").set_index("timestamp")
        # Regularize onto strict 5-min grid
        resampled = g["glucose"].resample("5min").mean()
        # Impute up to 3 missing samples (15 mins), leave longer gaps as NaN
        imputed = resampled.interpolate(method="time", limit=3)
        
        times = imputed.index
        vals = imputed.values
        
        X_list, y_list, base_list, meta_list = [], [], [], []
        
        for i in range(past_steps, len(vals) - ahead_steps + 1):
            window_vals = vals[i - past_steps : i]
            target_val = vals[i + ahead_steps - 1]
            target_time = times[i + ahead_steps - 1]
            curr_time = times[i - 1]
            
            # Skip if any NaN in window or target
            if np.isnan(window_vals).any() or np.isnan(target_val):
                continue
                
            curr_g = window_vals[-1]
            v15 = curr_g - window_vals[-4] # 15 min delta
            v30 = curr_g - window_vals[-7] # 30 min delta
            mean_1h = np.mean(window_vals)
            std_1h = np.std(window_vals)
            min_1h = np.min(window_vals)
            max_1h = np.max(window_vals)
            
            hr = curr_time.hour + curr_time.minute / 60.0
            hr_sin = np.sin(2 * np.pi * hr / 24.0)
            hr_cos = np.cos(2 * np.pi * hr / 24.0)
            
            features = list(window_vals) + [v15, v30, mean_1h, std_1h, min_1h, max_1h, hr_sin, hr_cos]
            
            X_list.append(features)
            y_list.append(target_val)
            base_list.append(curr_g) # Persistence baseline: future = current
            meta_list.append({
                "patient_id": pid,
                "timestamp": curr_time.isoformat(),
                "target_timestamp": target_time.isoformat()
            })
            
        pdf = pd.DataFrame(X_list, columns=FEATURE_NAMES)
        pdf["target"] = y_list
        pdf["baseline"] = base_list
        meta_df = pd.DataFrame(meta_list)
        combined = pd.concat([meta_df, pdf], axis=1)
        patient_dfs.append(combined)
        
    return pd.concat(patient_dfs, ignore_index=True)

def train_and_evaluate():
    data_path = "glucobloom/data/cgm_multi_patient.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing, resampling, and windowing...")
    dataset = preprocess_and_window(df)
    print(f"Constructed {len(dataset)} valid sliding feature windows across {dataset['patient_id'].nunique()} patients.")
    
    # Strict Patient-Level Splitting
    train_patients = ["p1", "p2", "p3", "p4", "p5", "p6"]
    val_patients = ["p7"]
    test_patients = ["p8", "p9", "p10"]
    
    train_df = dataset[dataset["patient_id"].isin(train_patients)]
    val_df = dataset[dataset["patient_id"].isin(val_patients)]
    test_df = dataset[dataset["patient_id"].isin(test_patients)]
    
    X_train, y_train = train_df[FEATURE_NAMES].values, train_df["target"].values
    X_val, y_val = val_df[FEATURE_NAMES].values, val_df["target"].values
    X_test, y_test = test_df[FEATURE_NAMES].values, test_df["target"].values
    base_test = test_df["baseline"].values
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Unseen Test={len(test_df)}")
    
    # 1. Persistence Baseline
    base_mae = mean_absolute_error(y_test, base_test)
    base_rmse = root_mean_squared_error(y_test, base_test)
    print(f"-> Persistence Baseline (Future = Current) on Unseen Patients: MAE={base_mae:.2f} mg/dL, RMSE={base_rmse:.2f} mg/dL")
    
    # 2. Linear Regression Benchmark
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = root_mean_squared_error(y_test, lr_pred)
    print(f"-> Linear Regression Benchmark: MAE={lr_mae:.2f} mg/dL, RMSE={lr_rmse:.2f} mg/dL")
    
    # 3. XGBoost General Foundation Model
    print("Training XGBoost General Foundation Model...")
    xgb_general = XGBRegressor(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_general.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_pred = xgb_general.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = root_mean_squared_error(y_test, xgb_pred)
    improvement = ((base_mae - xgb_mae) / base_mae) * 100.0
    print(f"-> XGBoost General Model: MAE={xgb_mae:.2f} mg/dL, RMSE={xgb_rmse:.2f} mg/dL ({improvement:.1f}% improvement over baseline)")
    
    # 4. Quantile Regression for Uncertainty (80% Confidence Interval: 10th & 90th percentiles)
    print("Training Quantile Regressors for 80% Prediction Interval...")
    xgb_lo = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.10,
        n_estimators=120,
        learning_rate=0.06,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    xgb_lo.fit(X_train, y_train)
    
    xgb_hi = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        n_estimators=120,
        learning_rate=0.06,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    xgb_hi.fit(X_train, y_train)
    
    lo_preds = xgb_lo.predict(X_test)
    hi_preds = xgb_hi.predict(X_test)
    
    # Check empirical coverage
    in_range = (y_test >= lo_preds) & (y_test <= hi_preds)
    coverage = np.mean(in_range) * 100.0
    print(f"-> Quantile 80% Interval Empirical Coverage on Unseen Patients: {coverage:.1f}% (Nominal target: 80.0%)")
    
    # 5. Digital Twin Personalization on Unseen Patient p8
    print("\n--- Digital Twin Personalization Experiment (Patient p8) ---")
    p8_df = dataset[dataset["patient_id"] == "p8"].sort_values("timestamp").reset_index(drop=True)
    # Total ~21 days. Days 1-14: adaptation stages. Days 15-21: fixed temporal test evaluation holdout
    steps_per_day = 288
    holdout_start_idx = 14 * steps_per_day
    
    p8_eval = p8_df.iloc[holdout_start_idx:].copy()
    X_p8_eval = p8_eval[FEATURE_NAMES].values
    y_p8_eval = p8_eval["target"].values
    
    # General model performance on p8 holdout
    p8_gen_pred = xgb_general.predict(X_p8_eval)
    p8_gen_mae = mean_absolute_error(y_p8_eval, p8_gen_pred)
    print(f"Patient p8 - General Model Baseline MAE on Days 15-21: {p8_gen_mae:.2f} mg/dL")
    
    evolution_records = []
    adaptation_days = [1, 3, 7, 14]
    
    best_personal_model = None
    
    for days in adaptation_days:
        adapt_end_idx = days * steps_per_day
        adapt_data = p8_df.iloc[:adapt_end_idx]
        X_adapt, y_adapt = adapt_data[FEATURE_NAMES].values, adapt_data["target"].values
        
        # Fine-tune the general booster with individual's data
        personal_model = XGBRegressor(
            n_estimators=50,
            learning_rate=0.03,
            max_depth=4,
            random_state=42
        )
        personal_model.fit(X_adapt, y_adapt, xgb_model=xgb_general.get_booster())
        
        personal_pred = personal_model.predict(X_p8_eval)
        personal_mae = mean_absolute_error(y_p8_eval, personal_pred)
        
        evolution_records.append({
            "adaptation_days": days,
            "samples_seen": len(adapt_data),
            "general_model_mae": round(float(p8_gen_mae), 2),
            "personalized_twin_mae": round(float(personal_mae), 2),
            "improvement_pct": round(float(((p8_gen_mae - personal_mae) / p8_gen_mae) * 100.0), 1)
        })
        print(f"  -> Twin after {days:2d} days ({len(adapt_data)} samples): MAE={personal_mae:.2f} mg/dL (Gain: +{evolution_records[-1]['improvement_pct']}%)")
        best_personal_model = personal_model
        
    # 6. Save Models & Artifacts
    os.makedirs("glucobloom/model", exist_ok=True)
    os.makedirs("glucobloom/outputs", exist_ok=True)
    
    joblib.dump(xgb_general, "glucobloom/model/glucobloom_general.pkl")
    joblib.dump(xgb_lo, "glucobloom/model/glucobloom_lo.pkl")
    joblib.dump(xgb_hi, "glucobloom/model/glucobloom_hi.pkl")
    joblib.dump(best_personal_model, "glucobloom/model/glucobloom_personalized_p8.pkl")
    print("Saved model artifacts (.pkl files) successfully.")
    
    # Save Feature Importance for Explainable AI
    importances = xgb_general.feature_importances_
    xai_dict = {
        name: round(float(imp), 4) for name, imp in zip(FEATURE_NAMES, importances)
    }
    sorted_xai = dict(sorted(xai_dict.items(), key=lambda item: item[1], reverse=True))
    with open("glucobloom/outputs/explainability.json", "w") as f:
        json.dump(sorted_xai, f, indent=2)
        
    # Save Evolution JSON
    with open("glucobloom/outputs/evolution.json", "w") as f:
        json.dump(evolution_records, f, indent=2)
        
    # Save Time Machine payload for frontend slider (e.g., 200 consecutive steps from p8 test holdout)
    time_slice = p8_eval.iloc[:200].copy()
    slice_X = time_slice[FEATURE_NAMES].values
    slice_preds = best_personal_model.predict(slice_X)
    slice_lo = xgb_lo.predict(slice_X)
    slice_hi = xgb_hi.predict(slice_X)
    
    time_machine_data = []
    for idx in range(len(time_slice)):
        row = time_slice.iloc[idx]
        time_machine_data.append({
            "patient_id": "p8",
            "timestamp": row["timestamp"],
            "target_timestamp": row["target_timestamp"],
            "current_glucose": round(float(row["g_11"]), 1),
            "actual_glucose_30m": round(float(row["target"]), 1),
            "predicted_glucose_30m": round(float(slice_preds[idx]), 1),
            "lower_bound_80": round(float(slice_lo[idx]), 1),
            "upper_bound_80": round(float(slice_hi[idx]), 1),
            "error_mgdl": round(float(abs(row["target"] - slice_preds[idx])), 1)
        })
        
    with open("glucobloom/outputs/time_machine.json", "w") as f:
        json.dump(time_machine_data, f, indent=2)
        
    # Save overall evaluation summary
    results_summary = {
        "persistence_baseline": {"mae": round(float(base_mae), 2), "rmse": round(float(base_rmse), 2)},
        "linear_regression": {"mae": round(float(lr_mae), 2), "rmse": round(float(lr_rmse), 2)},
        "xgboost_general": {"mae": round(float(xgb_mae), 2), "rmse": round(float(xgb_rmse), 2)},
        "quantile_coverage_pct": round(float(coverage), 1),
        "personalization_p8_final_mae": round(float(evolution_records[-1]["personalized_twin_mae"]), 2),
        "personalization_improvement_pct": evolution_records[-1]["improvement_pct"]
    }
    with open("glucobloom/outputs/evaluation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print("\nTraining and artifact generation complete!")

if __name__ == "__main__":
    train_and_evaluate()
