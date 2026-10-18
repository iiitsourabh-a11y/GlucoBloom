"""
Clarke Error Grid Analysis (EGA) for Blood Glucose Forecasting
Zones:
- Zone A: Clinically accurate (within 20% or both <= 70 mg/dL)
- Zone B: Benign errors (would lead to benign or no treatment)
- Zone C: Overcorrection errors (would lead to unnecessary treatment)
- Zone D: Dangerous failure to detect hypo/hyperglycemia
- Zone E: Erroneous treatment (confuses hypo with hyper)
"""

import json
import os
import numpy as np
import pandas as pd

def assign_clarke_zone(ref, pred):
    """
    Assigns Clarke Error Grid zone for reference (actual) vs predicted glucose.
    ref, pred are in mg/dL.
    """
    # Zone A
    if (pred <= 70 and ref <= 70) or \
       (pred <= 1.2 * ref and pred >= 0.8 * ref):
        return "A"
    
    # Zone E
    if (ref <= 70 and pred >= 180) or (ref >= 180 and pred <= 70):
        return "E"
    
    # Zone D
    if (ref >= 240 and 70 <= pred <= 180) or \
       (ref <= 70 and 70 <= pred <= 180):
        return "D"
    
    # Zone C
    if (70 <= ref <= 290 and pred >= ref + 110) or \
       (130 <= ref <= 180 and pred <= (7/5) * ref - 182):
        return "C"
    
    # Zone B: All other benign errors
    return "B"

def compute_clarke_statistics(actual_vals, pred_vals):
    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    total = len(actual_vals)
    
    zone_assignments = []
    for r, p in zip(actual_vals, pred_vals):
        z = assign_clarke_zone(r, p)
        zones[z] += 1
        zone_assignments.append(z)
        
    percentages = {z: round((count / total) * 100.0, 2) for z, count in zones.items()}
    clinically_acceptable = round(percentages["A"] + percentages["B"], 2)
    
    return {
        "zone_counts": zones,
        "zone_percentages": percentages,
        "clinically_acceptable_A_plus_B_pct": clinically_acceptable,
        "total_evaluated_points": total
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tm_path = os.path.join(base_dir, "outputs", "time_machine.json")
    
    if os.path.exists(tm_path):
        with open(tm_path, "r") as f:
            tm_data = json.load(f)
            
        actuals = [x["actual_glucose_30m"] for x in tm_data]
        preds = [x["predicted_glucose_30m"] for x in tm_data]
        
        results = compute_clarke_statistics(actuals, preds)
        print("Clarke Error Grid Results on Time Machine Holdout:")
        print(f"  Zone A (Accurate): {results['zone_percentages']['A']}%")
        print(f"  Zone B (Benign):   {results['zone_percentages']['B']}%")
        print(f"  Total Clinically Acceptable (A + B): {results['clinically_acceptable_A_plus_B_pct']}%")
        
        out_path = os.path.join(base_dir, "outputs", "clarke_error_grid.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
