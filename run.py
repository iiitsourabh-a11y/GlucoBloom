"""
GlucoBloom Single Launcher Script
Runs the Streamlit Web Application and/or FastAPI Service
"""

import sys
import subprocess
import os

def run_website():
    print("Launching GlucoBloom Streamlit Web Dashboard...")
    app_path = os.path.join(os.path.dirname(__file__), "website", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8501"])

def run_api():
    print("Launching GlucoBloom FastAPI Server at http://127.0.0.1:8000 ...")
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000", "--reload"], cwd=os.path.dirname(__file__))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
    else:
        run_website()
