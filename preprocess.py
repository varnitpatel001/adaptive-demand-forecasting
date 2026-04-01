import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# -------------------- CONFIG --------------------
LAGS = [1, 2, 3, 7]
ROLL_WINDOWS = [3, 7, 14]


# -------------------- LOAD --------------------
def load_data(path):
    df = pd.read_csv(path)
    logging.info(f"Loaded data: {df.shape}")
    return df


# -------------------- CLEAN --------------------
def clean_data(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'sales'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


# -------------------- TIME FEATURES --------------------
def add_time_features(df):
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday

    # 🔥 Cyclical encoding (important)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    return df


# -------------------- LAG FEATURES --------------------
def add_lag_features(df):
    for lag in LAGS:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    return df


# -------------------- ROLLING FEATURES --------------------
def add_rolling_features(df):
    for w in ROLL_WINDOWS:
        # strictly past → shift(1)
        df[f'roll_mean_{w}'] = df['sales'].shift(1).rolling(window=w).mean()
        df[f'roll_std_{w}'] = df['sales'].shift(1).rolling(window=w).std()
    return df


# -------------------- FINALIZE --------------------
def finalize(df, dropna=True):
    if dropna:
        df = df.dropna()

    df = df.reset_index(drop=True)
    return df


# -------------------- TRAIN PIPELINE --------------------
def preprocess_data(path):
    df = load_data(path)

    df = clean_data(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    df = finalize(df, dropna=True)

    logging.info(f"Processed data shape: {df.shape}")
    return df


# -------------------- REALTIME PIPELINE --------------------
def preprocess_realtime(df):
    """
    Safe for streaming inference.
    Requires at least max(LAGS, ROLL_WINDOWS) history.
    """

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Only keep last row (latest prediction point)
    latest = df.tail(1)

    # 🔥 Handle missing values safely (cold start)
    latest = latest.fillna(method='ffill').fillna(method='bfill')

    return latest

def preprocess_realtime(df):
    """
    Safe for streaming inference.
    Requires at least max(LAGS, ROLL_WINDOWS) history.
    """

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Only keep last row (latest prediction point)
    latest = df.tail(1)

    # 🔥 Handle missing values safely (cold start)
    latest = latest.fillna(method='ffill').fillna(method='bfill')

    return latest


# -------------------- FEATURE LIST --------------------
def get_feature_columns(df):
    return [col for col in df.columns if col not in ['date', 'sales']]