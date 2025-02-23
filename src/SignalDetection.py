import numpy as np
from scipy.stats import norm  # Import for correct z-score calculation

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        total_signal = self.hits + self.misses
        return self.hits / total_signal if total_signal > 0 else 0.5  # Prevent division by zero

    def false_alarm_rate(self):
        total_noise = self.false_alarms + self.correct_rejections
        return self.false_alarms / total_noise if total_noise > 0 else 0.5  # Prevent division by zero

    def d_prime(self):
        return self.z_score(self.hit_rate()) - self.z_score(self.false_alarm_rate())

    def criterion(self):
        return -0.5 * (self.z_score(self.hit_rate()) + self.z_score(self.false_alarm_rate()))

    @staticmethod
    def z_score(rate):
        # Ensure rate is within valid range to avoid math errors
        rate = np.clip(rate, 1e-6, 1 - 1e-6)
        return norm.ppf(rate)  # Correct way to compute z-score
