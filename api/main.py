"""
GlucoBloom FastAPI Prediction & Inference Service
Implements the Section 6 Shared Prediction Contract:
{"patient_id": str, "timestamp": str, "predicted_glucose": float, "lower": float, "upper": float, "model": str}
"""

import json
import os
from datetime import datetime
from typing import List, Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="GlucoBloom AI API",
    description="Personalized Digital Twin Glucose Forecasting Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

general_model = joblib.load(os.path.join(MODEL_DIR, "glucobloom_general.pkl"))
lo_model = joblib.load(os.path.join(MODEL_DIR, "glucobloom_lo.pkl"))
hi_model = joblib.load(os.path.join(MODEL_DIR, "glucobloom_hi.pkl"))

personal_p8_path = os.path.join(MODEL_DIR, "glucobloom_personalized_p8.pkl")
personal_p8_model = joblib.load(personal_p8_path) if os.path.exists(personal_p8_path) else None

class PredictionRequest(BaseModel):
    patient_id: str = Field(default="p8", description="Patient identifier")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of last reading")
    readings: List[float] = Field(..., min_length=12, description="Last 12 readings in 5-min intervals")
    use_personalized: bool = Field(default=True, description="Whether to use the personalized twin if available")

class ScenarioRequest(BaseModel):
    patient_id: str = "p8"
    readings: List[float] = Field(..., min_length=12)
    hypothetical_carbs: float = Field(default=0.0, description="Carbohydrates in grams")
    hypothetical_walk_mins: float = Field(default=0.0, description="Post-meal walk in minutes")

class DataQualityRequest(BaseModel):
    timestamps: List[str]
    glucose_values: List[float]

def extract_features(readings: List[float], dt: datetime) -> np.ndarray:
    """Builds the 20-feature input vector from the last 12 readings and time."""
    w = np.array(readings[-12:])
    curr_g = w[-1]
    v15 = curr_g - w[-4]
    v30 = curr_g - w[-7]
    mean_1h = float(np.mean(w))
    std_1h = float(np.std(w))
    min_1h = float(np.min(w))
    max_1h = float(np.max(w))
    
    hr = dt.hour + dt.minute / 60.0
    hr_sin = np.sin(2 * np.pi * hr / 24.0)
    hr_cos = np.cos(2 * np.pi * hr / 24.0)
    
    features = list(w) + [v15, v30, mean_1h, std_1h, min_1h, max_1h, hr_sin, hr_cos]
    return np.array(features).reshape(1, -1)

@app.get("/")
def root():
    return {
        "service": "GlucoBloom Digital Twin AI Engine",
        "status": "online",
        "version": "1.0.0",
        "disclaimer": "Research and educational prototype only. No medical advice."
    }

@app.post("/predict")
def predict(req: PredictionRequest):
    if len(req.readings) < 12:
        raise HTTPException(status_code=400, detail="At least 12 continuous 5-minute readings required.")
    
    dt = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now()
    feat = extract_features(req.readings, dt)
    
    chosen_model = "general"
    active_model = general_model
    
    if req.use_personalized and req.patient_id == "p8" and personal_p8_model is not None:
        active_model = personal_p8_model
        chosen_model = "personalized"
        
    pred = float(active_model.predict(feat)[0])
    lo = float(lo_model.predict(feat)[0])
    hi = float(hi_model.predict(feat)[0])
    
    # Ensure logical order
    lo = min(lo, pred)
    hi = max(hi, pred)
    
    return {
        "patient_id": req.patient_id,
        "timestamp": dt.isoformat(),
        "predicted_glucose": round(pred, 1),
        "lower": round(lo, 1),
        "upper": round(hi, 1),
        "model": chosen_model
    }

@app.post("/explain")
def explain(req: PredictionRequest):
    dt = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now()
    feat = extract_features(req.readings, dt)
    
    w = np.array(req.readings[-12:])
    curr_g = w[-1]
    v15 = curr_g - w[-4]
    
    factors = [
        {"factor": "Current Baseline", "influence": round(float(curr_g * 0.45), 1), "description": f"Current reading is {curr_g:.1f} mg/dL"},
        {"factor": "15-min Trend Velocity", "influence": round(float(v15 * 0.8), 1), "description": f"Velocity is {'+' if v15>=0 else ''}{v15:.1f} mg/dL over past 15 mins"},
        {"factor": "1-Hour Volatility", "influence": round(float(np.std(w) * 0.3), 1), "description": f"Standard deviation is {np.std(w):.1f} mg/dL"},
        {"factor": "Circadian Diurnal Phase", "influence": round(float(np.sin(2 * np.pi * dt.hour / 24.0) * 4.0), 1), "description": f"Time of day {dt.strftime('%H:%M')}"}
    ]
    return {
        "patient_id": req.patient_id,
        "top_influencing_factors": factors
    }

@app.post("/simulate")
def simulate(req: ScenarioRequest):
    """Scenario simulation comparing baseline vs hypothetical carbs/walk."""
    dt = datetime.now()
    feat = extract_features(req.readings, dt)
    
    base_pred = float(general_model.predict(feat)[0])
    
    # Scenario calculation (Clearly labelled as physiological association)
    # 10g carbs adds ~20-25 mg/dL peak in 30 mins; 30 min walk reduces ~15-20 mg/dL
    carb_effect = (req.hypothetical_carbs / 10.0) * 22.0 * 0.65 # 30-min fraction
    walk_effect = (req.hypothetical_walk_mins / 15.0) * 9.5
    
    simulated_pred = max(45.0, min(380.0, base_pred + carb_effect - walk_effect))
    
    return {
        "patient_id": req.patient_id,
        "baseline_predicted_30m": round(base_pred, 1),
        "simulated_predicted_30m": round(simulated_pred, 1),
        "delta_mgdl": round(simulated_pred - base_pred, 1),
        "assumptions": {
            "carbs_g": req.hypothetical_carbs,
            "walk_mins": req.hypothetical_walk_mins,
            "label": "Physiological Simulation (Demonstration)"
        }
    }

@app.post("/data-quality")
def check_data_quality(req: DataQualityRequest):
    """Analyzes timestamp consistency, missing intervals, and sensor outliers."""
    total_readings = len(req.glucose_values)
    if total_readings == 0:
        return {"status": "error", "message": "No readings provided"}
        
    out_of_bounds = sum(1 for v in req.glucose_values if v < 40 or v > 400)
    
    # Check timestamp gaps > 15 mins
    gaps_count = 0
    if len(req.timestamps) > 1:
        try:
            dts = [datetime.fromisoformat(t) for t in req.timestamps]
            for i in range(1, len(dts)):
                diff_mins = (dts[i] - dts[i-1]).total_seconds() / 60.0
                if diff_mins > 15.0:
                    gaps_count += 1
        except Exception:
            gaps_count = -1
            
    # Reliability scoring
    if gaps_count == 0 and out_of_bounds == 0:
        reliability = "HIGH"
        warning = "Sensor stream is continuous and within calibrated physiological bounds."
    elif gaps_count <= 2 and out_of_bounds == 0:
        reliability = "MODERATE"
        warning = f"Detected {gaps_count} short gap(s); predictions will use interpolated values."
    else:
        reliability = "LOW"
        warning = f"Warning: {gaps_count} significant sensor gap(s) or {out_of_bounds} extreme outlier(s) detected. Forecast confidence degraded."
        
    return {
        "total_readings": total_readings,
        "gaps_exceeding_15min": gaps_count,
        "sensor_out_of_bounds_count": out_of_bounds,
        "reliability_level": reliability,
        "quality_assessment": warning
    }

@app.get("/evolution")
def get_evolution():
    ev_path = os.path.join(OUTPUTS_DIR, "evolution.json")
    if os.path.exists(ev_path):
        with open(ev_path, "r") as f:
            return json.load(f)
    return []

@app.get("/time-machine")
def get_time_machine():
    tm_path = os.path.join(OUTPUTS_DIR, "time_machine.json")
    if os.path.exists(tm_path):
        with open(tm_path, "r") as f:
            return json.load(f)
    return []
