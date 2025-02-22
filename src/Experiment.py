import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

class Experiment:
    def __init__(self):
        """Initialize an empty list to store SignalDetection objects and their labels."""
        self.conditions = []

    def add_condition(self, sdt_obj, label=None):
        """Adds a SignalDetection object and an optional label to the experiment."""
        if not isinstance(sdt_obj, SignalDetection):
            raise TypeError("sdt_obj must be an instance of SignalDetection")
        self.conditions.append((sdt_obj, label))

    def sorted_roc_points(self):
        """Returns sorted false alarm rates and hit rates for plotting the ROC curve."""
        if not self.conditions:
            raise ValueError("No conditions added to the experiment.")

        roc_points = [(cond.false_alarm_rate(), cond.hit_rate()) for cond, _ in self.conditions] #Extract false alarm rates & hit rates

        roc_points.sort() #false alarm rate sorted

        #Unpack sorted values
        false_alarm_rates, hit_rates = zip(*roc_points)
        return list(false_alarm_rates), list(hit_rates)

    def compute_auc(self):
        """Computes the Area Under the Curve (AUC) for the stored conditions."""
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        return trapezoid(hit_rates, false_alarm_rates)

    def plot_roc_curve(self, show_plot=True):
        """Plots the ROC curve using the stored conditions."""
        false_alarm_rates, hit_rates = self.sorted_roc_points()

        plt.figure(figsize=(6, 6))
        plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', label="ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC=0.5)")
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("ROC Curve")
        plt.legend()
        
        if show_plot:
            plt.show()

        #Using the help of ai to organize
