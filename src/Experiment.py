import numpy as np
from SignalDetection import SignalDetection

class Experiment:
    def __init__(self):
        self.conditions = []
        self.labels = []

    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        self.conditions.append(sdt_obj)
        self.labels.append(label)

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        if not self.conditions:
            raise ValueError("No conditions added to the experiment.")
        
        false_alarm_rates = [sdt.false_alarm_rate() for sdt in self.conditions]
        hit_rates = [sdt.hit_rate() for sdt in self.conditions]
        
        sorted_indices = np.argsort(false_alarm_rates).tolist()
        sorted_false_alarm_rates = [false_alarm_rates[i] for i in sorted_indices]
        sorted_hit_rates = [hit_rates[i] for i in sorted_indices]
        
        return sorted_false_alarm_rates, sorted_hit_rates

    def compute_auc(self) -> float:
        if not self.conditions:
            raise ValueError("No conditions have been added to the experiment.")
        
        far, hr = self.sorted_roc_points()

        # Ensure (0,0) and (1,1) are included
        far = [0] + far + [1]
        hr = [0] + hr + [1]
        
        # Compute AUC using NumPy's trapezoidal rule
        auc = np.trapz(hr, far)
        
        return auc

    def plot_roc_curve(self, show_plot: bool = True):
        # This method is optional and not part of the grading
        import matplotlib.pyplot as plt
        
        far, hr = self.sorted_roc_points()
        plt.plot(far, hr, 'b-', label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--', label='Chance Level')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        if show_plot:
            plt.show()
