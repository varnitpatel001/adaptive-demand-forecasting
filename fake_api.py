import numpy as np

def get_actual_value(pred, step):
    # 1. Trend (slow drift over time)
    trend = 0.05 * step

    # 2. Seasonality (periodic demand cycles)
    seasonality = 20 * np.sin(2 * np.pi * step / 24)   # daily cycle

    # 3. Noise (controlled randomness)
    noise = np.random.normal(0, 15)

    # 4. Sudden shock (rare events)
    shock = 0
    if np.random.rand() < 0.10:  # 5% chance
        shock = np.random.normal(0, 80)

    # 5. Combine everything
    actual = pred + trend + seasonality + noise + shock

    return actual