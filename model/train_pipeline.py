"""
GlucoBloom Training & Digital Twin Personalization Pipeline
Trained on 200 Virtual Patients (806,400 readings) with CGM, Meals, Activity & Sleep.
Evaluated strictly on Unseen Test Patients (175 - 199).
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "timeseries.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FEATURE_COLS = [
    "cgm", "cgm_lag_1", "cgm_lag_2", "cgm_lag_3", "cgm_lag_6", "cgm_lag_12",
    "velocity_15m", "velocity_30m", "carbs_last_30m", "steps_last_30m",
    "hr", "hrv", "asleep", "hour_sin", "hour_cos"
]

def load_and_feature_engineer(sample_patients: int = 200):
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    
    if sample_patients < 200:
        df = df[df["patient_id"] < sample_patients].copy()
        
    df = df.sort_values(["patient_id", "minute"]).reset_index(drop=True)
    grouped = df.groupby("patient_id", sort=False)
    
    # Target: CGM 30 mins ahead (6 steps)
    df["target"] = grouped["cgm"].shift(-6)
    
    # Historical Lags
    for lag in [1, 2, 3, 6, 12]:
        df[f"cgm_lag_{lag}"] = grouped["cgm"].shift(lag)
        
    # Velocity features
    df["velocity_15m"] = df["cgm"] - df["cgm_lag_3"]
    df["velocity_30m"] = df["cgm"] - df["cgm_lag_6"]
    
    # Cumulative recent lifestyle signals
    df["carbs_last_30m"] = grouped["carbs_g"].rolling(window=6, min_periods=1).sum().reset_index(drop=True)
    df["steps_last_30m"] = grouped["steps"].rolling(window=6, min_periods=1).sum().reset_index(drop=True)
    
    # Circadian sin/cos features
    hr = df["hour"]
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
    
    # Drop rows with NaN (due to lags, dropouts, or forward target shift)
    clean_df = df.dropna(subset=FEATURE_COLS + ["target"]).copy().reset_index(drop=True)
    return clean_df

def run_pipeline():
    df = load_and_feature_engineer()
    print(f"Dataset clean and engineered: {len(df):,} valid samples across {df['patient_id'].nunique()} patients.")
    
    # Strict Patient Split:
    # Train: patients 0 to 159 (160 patients)
    # Val: patients 160 to 174 (15 patients)
    # Unseen Test: patients 175 to 199 (25 patients)
    train_df = df[df["patient_id"] < 160]
    val_df = df[(df["patient_id"] >= 160) & (df["patient_id"] < 175)]
    test_df = df[df["patient_id"] >= 175]
    
    print(f"Splits: Train={len(train_df):,} | Val={len(val_df):,} | Unseen Test={len(test_df):,}")
    
    X_train, y_train = train_df[FEATURE_COLS].values, train_df["target"].values
    X_val, y_val = val_df[FEATURE_COLS].values, val_df["target"].values
    X_test, y_test = test_df[FEATURE_COLS].values, test_df["target"].values
    
    # 1. Baseline: Future CGM = Current CGM
    base_preds = test_df["cgm"].values
    base_mae = mean_absolute_error(y_test, base_preds)
    base_rmse = root_mean_squared_error(y_test, base_preds)
    print(f"-> Persistence Baseline (Future = Current) on Unseen Patients: MAE={base_mae:.2f} mg/dL, RMSE={base_rmse:.2f} mg/dL")
    
    # 2. Linear Regression Benchmark
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = root_mean_squared_error(y_test, lr_preds)
    print(f"-> Linear Regression Benchmark: MAE={lr_mae:.2f} mg/dL, RMSE={lr_rmse:.2f} mg/dL")
    
    # 3. XGBoost General Foundation Model
    print("Training XGBoost General Model on 160 patients...")
    xgb_gen = XGBRegressor(
        n_estimators=160,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_gen.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_preds = xgb_gen.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = root_mean_squared_error(y_test, xgb_preds)
    impr = ((base_mae - xgb_mae) / base_mae) * 100.0
    print(f"-> XGBoost General Model: MAE={xgb_mae:.2f} mg/dL, RMSE={xgb_rmse:.2f} mg/dL ({impr:.1f}% gain over baseline)")
    
    # 4. Quantile Regression for 80% Prediction Cone
    print("Training Quantile Regressors (10th and 90th percentiles)...")
    xgb_lo = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.10,
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    xgb_lo.fit(X_train, y_train)
    
    xgb_hi = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    xgb_hi.fit(X_train, y_train)
    
    lo_preds = xgb_lo.predict(X_test)
    hi_preds = xgb_hi.predict(X_test)
    
    coverage = np.mean((y_test >= lo_preds) & (y_test <= hi_preds)) * 100.0
    print(f"-> Quantile 80% Empirical Coverage on Unseen Test Patients: {coverage:.1f}%")
    
    # 5. Digital Twin Personalization on Unseen Patient 175
    print("\n--- Digital Twin Personalization Experiment (Patient 175) ---")
    p_data = df[df["patient_id"] == 175].sort_values("minute").reset_index(drop=True)
    # Days 0 to 9: adaptation. Days 10 to 13: held-out evaluation
    eval_holdout = p_data[p_data["day"] >= 10].copy()
    X_p_eval = eval_holdout[FEATURE_COLS].values
    y_p_eval = eval_holdout["target"].values
    
    p_gen_preds = xgb_gen.predict(X_p_eval)
    p_gen_mae = mean_absolute_error(y_p_eval, p_gen_preds)
    print(f"Patient 175 - General Model MAE on Holdout Days 10-13: {p_gen_mae:.2f} mg/dL")
    
    evolution_records = []
    adaptation_day_cutoffs = [1, 3, 5, 8]
    best_personal_model = None
    
    for days in adaptation_day_cutoffs:
        adapt_subset = p_data[p_data["day"] < days]
        X_ad = adapt_subset[FEATURE_COLS].values
        y_ad = adapt_subset["target"].values
        
        # Fine-tune general booster
        pers_model = XGBRegressor(
            n_estimators=40,
            learning_rate=0.03,
            max_depth=4,
            random_state=42
        )
        pers_model.fit(X_ad, y_ad, xgb_model=xgb_gen.get_booster())
        
        p_pers_preds = pers_model.predict(X_p_eval)
        p_pers_mae = mean_absolute_error(y_p_eval, p_pers_preds)
        gain = ((p_gen_mae - p_pers_mae) / p_gen_mae) * 100.0
        
        evolution_records.append({
            "adaptation_days": days,
            "samples_seen": len(adapt_subset),
            "general_model_mae": round(float(p_gen_mae), 2),
            "personalized_twin_mae": round(float(p_pers_mae), 2),
            "improvement_pct": round(float(gain), 1)
        })
        print(f"  -> Twin after {days} days ({len(adapt_subset)} samples): MAE={p_pers_mae:.2f} mg/dL (Gain: +{gain:.1f}%)")
        best_personal_model = pers_model
        
    # 6. Save models and outputs
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    joblib.dump(xgb_gen, os.path.join(MODEL_DIR, "glucobloom_general.pkl"))
    joblib.dump(xgb_lo, os.path.join(MODEL_DIR, "glucobloom_lo.pkl"))
    joblib.dump(xgb_hi, os.path.join(MODEL_DIR, "glucobloom_hi.pkl"))
    joblib.dump(best_personal_model, os.path.join(MODEL_DIR, "glucobloom_personalized_p8.pkl"))
    
    # Save Time Machine payload from Patient 175 holdout
    slice_df = eval_holdout.iloc[100:350].copy()
    slice_X = slice_df[FEATURE_COLS].values
    slice_preds = best_personal_model.predict(slice_X)
    slice_lo = xgb_lo.predict(slice_X)
    slice_hi = xgb_hi.predict(slice_X)
    
    time_machine_data = []
    base_dt = pd.to_datetime("2026-10-10 00:00:00")
    for i in range(len(slice_df)):
        r = slice_df.iloc[i]
        sim_time = base_dt + pd.Timedelta(minutes=int(r["minute"]))
        target_time = sim_time + pd.Timedelta(minutes=30)
        time_machine_data.append({
            "patient_id": "p175",
            "timestamp": sim_time.isoformat(),
            "target_timestamp": target_time.isoformat(),
            "current_glucose": round(float(r["cgm"]), 1),
            "actual_glucose_30m": round(float(r["target"]), 1),
            "predicted_glucose_30m": round(float(slice_preds[i]), 1),
            "lower_bound_80": round(float(slice_lo[i]), 1),
            "upper_bound_80": round(float(slice_hi[i]), 1),
            "error_mgdl": round(float(abs(r["target"] - slice_preds[i])), 1)
        })
        
    with open(os.path.join(OUTPUTS_DIR, "time_machine.json"), "w") as f:
        json.dump(time_machine_data, f, indent=2)
        
    with open(os.path.join(OUTPUTS_DIR, "evolution.json"), "w") as f:
        json.dump(evolution_records, f, indent=2)
        
    # Feature importances
    imps = {name: round(float(imp), 4) for name, imp in zip(FEATURE_COLS, xgb_gen.feature_importances_)}
    sorted_imps = dict(sorted(imps.items(), key=lambda item: item[1], reverse=True))
    with open(os.path.join(OUTPUTS_DIR, "explainability.json"), "w") as f:
        json.dump(sorted_imps, f, indent=2)
        
    summary = {
        "persistence_baseline": {"mae": round(float(base_mae), 2), "rmse": round(float(base_rmse), 2)},
        "linear_regression": {"mae": round(float(lr_mae), 2), "rmse": round(float(lr_rmse), 2)},
        "xgboost_general": {"mae": round(float(xgb_mae), 2), "rmse": round(float(xgb_rmse), 2)},
        "quantile_coverage_pct": round(float(coverage), 1),
        "personalization_p175_final_mae": round(float(evolution_records[-1]["personalized_twin_mae"]), 2),
        "personalization_improvement_pct": evolution_records[-1]["improvement_pct"]
    }
    with open(os.path.join(OUTPUTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\nTraining on 200-patient dataset completed successfully!")

if __name__ == "__main__":
    run_pipeline()
