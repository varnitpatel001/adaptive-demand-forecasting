from collections import deque
import numpy as np

class DriftDetector:

    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 2.0,
        min_std: float = 1e-6,
        confirm_drift: bool = False,
        confirm_window: int = 2,
    ):
        self.errors = deque(maxlen=window_size)
        self.threshold = threshold
        self.min_std = min_std

        # Drift confirmation mechanism
        self.confirm_drift = confirm_drift
        self.confirm_window = confirm_window
        self._drift_count = 0

    def update(self, actual: float, predicted: float) -> bool:
      
        error = abs(actual - predicted)
        self.errors.append(error)

        # Not enough data yet
        if len(self.errors) < self.errors.maxlen:
            return False

        errors_array = np.array(self.errors)

        mean = errors_array.mean()
        std = errors_array.std()
        std = max(std, self.min_std)

        recent_error = errors_array[-1]

        drift_signal = recent_error > (mean + self.threshold * std)

        # --- Drift confirmation logic ---
        if self.confirm_drift:
            if drift_signal:
                self._drift_count += 1
            else:
                self._drift_count = 0

            if self._drift_count >= self.confirm_window:
                self._drift_count = 0
                return True

            return False

        return drift_signal

    def reset(self):
      
        self.errors.clear()
        self._drift_count = 0

    def get_stats(self):
      
        if len(self.errors) == 0:
            return {}

        errors_array = np.array(self.errors)

        return {
            "mean_error": float(errors_array.mean()),
            "std_error": float(errors_array.std()),
            "latest_error": float(errors_array[-1]),
            "window_size": len(self.errors),
        }