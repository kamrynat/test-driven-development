import numpy as np
from scipy.stats import norm

class SignalDetection:  # used ZotGPT to make structure of class
    def __init__(self, hits, misses, false_alarms, correct_rejections): 
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections
        
    def hit_rate(self):
        return self.hits / (self.hits + self.misses)

    def false_alarm_rate(self):
        return self.false_alarms / (self.false_alarms + self.correct_rejections) 
    
    def d_prime(self): # used ZotGPT to write this function
        hit_rate = self.hit_rate()
        fa_rate = self.false_alarm_rate()
        
        # Handle edge cases
        hit_rate = np.clip(hit_rate, 0.00001, 0.99999)
        fa_rate = np.clip(fa_rate, 0.00001, 0.99999)
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return z_hit - z_fa
    
    def criterion(self): # used ZotGPT to write this function
        hit_rate = self.hit_rate()
        fa_rate = self.false_alarm_rate()
        
        # Handle edge cases
        hit_rate = np.clip(hit_rate, 0.00001, 0.99999)
        fa_rate = np.clip(fa_rate, 0.00001, 0.99999)
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        
        return -0.5 * (z_hit + z_fa)