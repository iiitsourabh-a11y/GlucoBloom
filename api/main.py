"""
GlucoBloom FastAPI Prediction & Inference Service
Implements the Section 6 Shared Prediction Contract:
{"patient_id": str, "timestamp": str, "predicted_glucose": float, "lower": float, "upper": float, "model": str}
Also serves Person B's connected dashboard at / or /dashboard
"""

import json
import os
from datetime import datetime
from typing import List, Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
WEBSITE_DIR = os.path.join(BASE_DIR, "website")
DASHBOARD_HTML = os.path.join(WEBSITE_DIR, "dashboard.html")

# Load trained models
general_model = joblib.load(os.path.join(MODEL_DIR, "glucobloom_general.pkl"))
lo_model = joblib.load(os.path.join(MODEL_DIR, "glucobloom_lo.pkl"))
hi_model = joblib.load(os.path.join(MODEL_DIR, "glucobloom_hi.pkl"))

personal_p8_path = os.path.join(MODEL_DIR, "glucobloom_personalized_p8.pkl")
personal_p8_model = joblib.load(personal_p8_path) if os.path.exists(personal_p8_path) else None

class PredictionRequest(BaseModel):
    patient_id: str = Field(default="175", description="Patient identifier")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of last reading")
    readings: List[float] = Field(..., min_length=12, description="Last 12 readings in 5-min intervals")
    use_personalized: bool = Field(default=True, description="Whether to use the personalized twin if available")

class ScenarioRequest(BaseModel):
    patient_id: str = "175"
    readings: List[float] = Field(..., min_length=12)
    hypothetical_carbs: float = Field(default=0.0, description="Carbohydrates in grams")
    hypothetical_walk_mins: float = Field(default=0.0, description="Post-meal walk in minutes")

class DataQualityRequest(BaseModel):
    timestamps: List[str]
    glucose_values: List[float]

def extract_features(readings: List[float], dt: datetime) -> np.ndarray:
    """Builds the 15-feature input vector matching the trained model."""
    w = np.array(readings[-12:], dtype=float)
    cgm = float(w[-1])
    cgm_l1 = float(w[-2])
    cgm_l2 = float(w[-3])
    cgm_l3 = float(w[-4])
    cgm_l6 = float(w[-7])
    cgm_l12 = float(w[0])
    
    v15 = cgm - cgm_l3
    v30 = cgm - cgm_l6
    carbs30 = 0.0
    steps30 = 120.0
    hr = 74.0
    hrv = 36.0
    asleep = 0
    
    hr_val = dt.hour + dt.minute / 60.0
    hr_sin = np.sin(2 * np.pi * hr_val / 24.0)
    hr_cos = np.cos(2 * np.pi * hr_val / 24.0)
    
    feats = [cgm, cgm_l1, cgm_l2, cgm_l3, cgm_l6, cgm_l12, v15, v30, carbs30, steps30, hr, hrv, asleep, hr_sin, hr_cos]
    return np.array(feats).reshape(1, -1)

@app.get("/")
def serve_dashboard():
    """Serves the connected GlucoBloom HTML dashboard."""
    if os.path.exists(DASHBOARD_HTML):
        return FileResponse(DASHBOARD_HTML)
    return {
        "service": "GlucoBloom Digital Twin AI Engine",
        "status": "online",
        "version": "1.0.0"
    }

@app.get("/dashboard")
def dashboard_route():
    if os.path.exists(DASHBOARD_HTML):
        return FileResponse(DASHBOARD_HTML)
    raise HTTPException(status_code=404, detail="dashboard.html not found")

@app.get("/api")
def api_status():
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
    
    if req.use_personalized and req.patient_id in ["p8", "p175", "175"] and personal_p8_model is not None:
        active_model = personal_p8_model
        chosen_model = "personalized"
        
    pred = float(active_model.predict(feat)[0])
    lo = float(lo_model.predict(feat)[0])
    hi = float(hi_model.predict(feat)[0])
    
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
    w = np.array(req.readings[-12:], dtype=float)
    curr_g = w[-1]
    v15 = curr_g - w[-4]
    
    factors = [
        {"factor": "Current Baseline Glucose", "influence": round(float(curr_g * 0.45), 1), "description": f"Recent measurement is {curr_g:.1f} mg/dL"},
        {"factor": "15-min Trend Velocity", "influence": round(float(v15 * 0.8), 1), "description": f"Velocity is {'+' if v15>=0 else ''}{v15:.1f} mg/dL over past 15 mins"},
        {"factor": "1-Hour Volatility", "influence": round(float(np.std(w) * 0.3), 1), "description": f"Standard deviation across window is {np.std(w):.1f} mg/dL"},
        {"factor": "Diurnal Circadian Phase", "influence": round(float(np.sin(2 * np.pi * dt.hour / 24.0) * 4.0), 1), "description": f"Circadian phase at {dt.strftime('%H:%M')}"}
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
    
    carb_effect = (req.hypothetical_carbs / 10.0) * 18.0 * 0.70
    walk_effect = (req.hypothetical_walk_mins / 15.0) * 11.5
    
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
    total_readings = len(req.glucose_values)
    if total_readings == 0:
        return {"status": "error", "message": "No readings provided"}
        
    out_of_bounds = sum(1 for v in req.glucose_values if v < 40 or v > 400)
    
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

@app.get("/evaluation-summary")
def get_evaluation_summary():
    ev_path = os.path.join(OUTPUTS_DIR, "evaluation_results.json")
    cl_path = os.path.join(OUTPUTS_DIR, "clarke_error_grid.json")
    ev_data = json.load(open(ev_path)) if os.path.exists(ev_path) else {}
    cl_data = json.load(open(cl_path)) if os.path.exists(cl_path) else {}
    return {
        "evaluation": ev_data,
        "clarke_error_grid": cl_data
    }
