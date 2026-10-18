from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api():
    print("Testing GET / ...")
    r = client.get("/")
    assert r.status_code == 200, r.text
    print("  Root response:", r.json())
    
    print("Testing POST /predict ...")
    sample_readings = [120.0, 122.5, 125.0, 128.0, 131.0, 134.5, 138.0, 142.0, 146.5, 151.0, 155.5, 160.0]
    payload = {
        "patient_id": "p8",
        "timestamp": "2026-10-18T12:00:00",
        "readings": sample_readings,
        "use_personalized": True
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    print("  Prediction Contract output:", r.json())
    
    print("Testing POST /explain ...")
    r = client.post("/explain", json=payload)
    assert r.status_code == 200, r.text
    print("  Explainability output:", r.json())
    
    print("Testing POST /simulate ...")
    sim_payload = {
        "patient_id": "p8",
        "readings": sample_readings,
        "hypothetical_carbs": 45.0,
        "hypothetical_walk_mins": 20.0
    }
    r = client.post("/simulate", json=sim_payload)
    assert r.status_code == 200, r.text
    print("  Simulate output:", r.json())
    
    print("Testing GET /evolution ...")
    r = client.get("/evolution")
    assert r.status_code == 200, r.text
    print(f"  Evolution records: {len(r.json())} stages")
    
    print("Testing GET /time-machine ...")
    r = client.get("/time-machine")
    assert r.status_code == 200, r.text
    print(f"  Time machine records: {len(r.json())} steps")
    
    print("\nALL API ENDPOINTS PASSED VERIFICATION!")

if __name__ == "__main__":
    test_api()
