import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------- CONFIG --------------------
DATA_PATH = "data/processed.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "outputs/metrics.json"

TEST_SIZE = 0.2
WINDOW_SIZE = None   # None = use full data (recommended)



# -------------------- UTIL --------------------
def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'sales'])
    df = df.sort_values("date")

    if WINDOW_SIZE:
        df = df.tail(WINDOW_SIZE)

    X = df.drop(columns=['sales', 'date'])
    y = df['sales']

    return X, y, df


# -------------------- MODEL --------------------
def build_model():
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )


# -------------------- TRAIN --------------------
def train_model():
    ensure_dirs()

    logging.info("Loading data...")
    df = pd.read_csv(DATA_PATH)

    X, y, df = prepare_data(df)

    # Time-based split
    split = int(len(df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = build_model()

    logging.info("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")

    package = {
        "model": model,
        "features": list(X.columns),
        "trained_at": datetime.now().isoformat(),
        "metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse)
        },
        "type": "initial"
    }

    joblib.dump(package, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(package["metrics"], f, indent=4)

    logging.info("Model saved.")
    return package


# -------------------- SAFE RETRAIN --------------------
def retrain_model(df_new):
    """
    Retrain model:
    - Uses ALL data
    - Gives higher weight to recent data
    - ALWAYS updates model (no rejection)
    """

    logging.info("Loading old model...")
    old_package = joblib.load(MODEL_PATH)
    old_metrics = old_package.get("metrics", {})

    # Merge old + new data
    df_old = pd.read_csv(DATA_PATH)
    df_all = pd.concat([df_old, df_new], ignore_index=True)

    # Prepare data
    X, y, df_all = prepare_data(df_all)

    # Time-based split
    split = int(len(df_all) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # -------------------- WEIGHTING --------------------
    def create_weights(n, strength=2.0):
        """
        Moderate recency bias (NOT aggressive)
        """
        w = np.linspace(1.0, strength, n)
        return w / w.mean()

    weights = create_weights(len(X_train))

    # -------------------- MODEL --------------------
    model = build_model()

    logging.info("Retraining with recency weighting...")
    model.fit(X_train, y_train, sample_weight=weights)

    # -------------------- EVALUATION --------------------
    preds = model.predict(X_test)

    new_mae = mean_absolute_error(y_test, preds)
    new_rmse = mean_squared_error(y_test, preds) ** 0.5

    logging.info(f"Old RMSE: {old_metrics.get('RMSE', 'N/A')}")
    logging.info(f"New MAE: {new_mae:.2f}")
    logging.info(f"New RMSE: {new_rmse:.2f}")

    # -------------------- SAVE ALWAYS --------------------
    package = {
        "model": model,
        "features": list(X.columns),
        "trained_at": datetime.now().isoformat(),
        "metrics": {
            "MAE": float(new_mae),
            "RMSE": float(new_rmse)
        },
        "type": "retrained"
    }

    joblib.dump(package, MODEL_PATH)

    logging.info("Model updated (forced retrain).")

    return package


# -------------------- RUN --------------------
if __name__ == "__main__":
    train_model()