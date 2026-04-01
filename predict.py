import pandas as pd


# ---------------- FEATURE BUILDER ----------------
def create_features(df):
    df = df.copy()

    # Lag features
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_14'] = df['sales'].shift(14)

    # Rolling
    df['rolling_mean_7'] = df['sales'].rolling(7).mean()
    df['rolling_std_7'] = df['sales'].rolling(7).std()

    # Time features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month

    df = df.dropna()

    return df


# ---------------- SINGLE PREDICTION ----------------
def predict_next(df, model, feature_names):
    df_features = create_features(df)

    latest_row = df_features.iloc[-1:]
    X = latest_row[feature_names]

    prediction = model.predict(X)[0]

    return prediction