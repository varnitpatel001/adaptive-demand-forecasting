import time
import logging
import pandas as pd
import joblib
from datetime import timedelta

from predict import predict_next
from drift import DriftDetector
from fake_api import get_actual_value
from utils import load_data, append_row
from train import retrain_model
import numpy as np
import json
from datetime import datetime
import os

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/processed.csv"

logging.basicConfig(level=logging.INFO)
OUTPUT_FILE = "outputs/live.json"


def save_output(pred, actual, drift, drift_count):
    os.makedirs("outputs", exist_ok=True)

    data = {
        "time": datetime.now().isoformat(),
        "predicted": float(pred),
        "actual": float(actual),
        "drift": bool(drift),
        "drift_count": int(drift_count)
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=4)


        



step = 0  # global or managed variable

def run_step(model):
    global step
    
    # 1. Generate prediction input (example)
    input_val = np.array([[5000]])  # or your real feature
    
    pred = model.predict(input_val)[0]
    
    # 2. Generate actual using pred + step
    actual = get_actual_value(pred, step)
    
    # 3. Drift logic
    drift = abs(pred - actual) > 300
    
    step += 1  # increment time 
    print(f"Step {step}: Pred={pred:.2f}, Actual={actual:.2f}, Drift={drift}")
  
    return {
        "predicted": pred,
        "actual": actual,
        "drift": drift
    }

def run_realtime():
    logging.info("Starting real-time system with auto-retraining...")

    # Load model
    package = joblib.load(MODEL_PATH)
    model = package["model"]
    features = package["features"]

    df = load_data(DATA_PATH)
    detector = DriftDetector()

    step = 0
    drift_count = 0

    while True:
        pred = predict_next(df, model, features)

        last_date = pd.to_datetime(df['date'].iloc[-1])
        next_date = last_date + timedelta(days=1)

        actual = get_actual_value(pred, step)

        logging.info(f"Pred: {pred:.2f}, Actual: {actual:.2f}")

        # Drift detection
        if detector.update(actual, pred):
            drift_count += 1
            logging.warning(f"⚠️ Drift detected! Count: {drift_count}")

        # 🔥 Retrain condition (important)
        if drift_count >= 4:
            logging.warning("🔄 Retraining model due to drift...")

            package = retrain_model(df)

            model = package["model"]
            features = package["features"]

            drift_count = 0  # reset after retrain
            print(df.count()) 
        # Update data
        df = append_row(df, next_date, actual)

        step += 1
        time.sleep(1)  # Simulate real-time delay
        save_output(pred, actual, drift_count > 0, drift_count)


