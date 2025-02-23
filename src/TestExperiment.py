import unittest
from Experiment import Experiment
from SignalDetection import SignalDetection

class TestExperiment(unittest.TestCase): # written using chatGPT

    def setUp(self):
        self.exp = Experiment()

    def test_add_condition(self):
        sdt = SignalDetection(15, 5, 7, 23)
        self.exp.add_condition(sdt, "Condition A")
        self.assertEqual(len(self.exp.conditions), 1)
        self.assertEqual(len(self.exp.labels), 1)
        self.assertEqual(self.exp.labels[0], "Condition A")

    def test_add_multiple_conditions(self):
        sdt1 = SignalDetection(15, 5, 7, 23)
        sdt2 = SignalDetection(12, 8, 5, 25)
        self.exp.add_condition(sdt1, "Condition A")
        self.exp.add_condition(sdt2, "Condition B")
        self.assertEqual(len(self.exp.conditions), 2)
        self.assertEqual(len(self.exp.labels), 2)
        self.assertEqual(self.exp.labels, ["Condition A", "Condition B"])

    def test_sorted_roc_points(self):
        sdt1 = SignalDetection(15, 5, 7, 23)  # FAR ≈ 0.2333, HR ≈ 0.75
        sdt2 = SignalDetection(12, 8, 5, 25)  # FAR ≈ 0.1667, HR ≈ 0.60
        self.exp.add_condition(sdt1, "Condition A")
        self.exp.add_condition(sdt2, "Condition B")
        far, hr = self.exp.sorted_roc_points()
        
        # Ensure values are correctly sorted
        self.assertAlmostEqual(far[0], min(0.2333, 0.1667), places=4)
        self.assertAlmostEqual(far[1], max(0.2333, 0.1667), places=4)
        self.assertAlmostEqual(hr[0], min(0.75, 0.60), places=4)
        self.assertAlmostEqual(hr[1], max(0.75, 0.60), places=4)

    def test_auc_empty(self):
        with self.assertRaises(ValueError):
            self.exp.compute_auc()

    def test_auc_two_points(self):
        sdt1 = SignalDetection(0, 10, 0, 10)  # (0, 0)
        sdt2 = SignalDetection(10, 0, 10, 0)  # (1, 1)
        self.exp.add_condition(sdt1, "Condition A")
        self.exp.add_condition(sdt2, "Condition B")
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.5, places=4)

    def test_auc_perfect(self):
        sdt1 = SignalDetection(0, 10, 0, 10)  # (0, 0)
        sdt2 = SignalDetection(10, 0, 0, 10)  # (0, 1)
        sdt3 = SignalDetection(10, 0, 10, 0)  # (1, 1)
        self.exp.add_condition(sdt1, "Condition A")
        self.exp.add_condition(sdt2, "Condition B")
        self.exp.add_condition(sdt3, "Condition C")
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 1.0, places=4)

    def test_auc_typical(self):
        sdt1 = SignalDetection(20, 10, 15, 25)  # (FAR = 0.375, HR = 0.667)
        sdt2 = SignalDetection(15, 15, 10, 30)  # (FAR = 0.250, HR = 0.500)
        self.exp.add_condition(sdt1, "Condition A")
        self.exp.add_condition(sdt2, "Condition B")
        auc = self.exp.compute_auc()
        # Expected AUC: 0.65625 based on the trapezoidal rule
        self.assertAlmostEqual(auc, 0.65625, places=4)

    def test_sorted_roc_points_empty(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()

    def test_zero_hit_rate(self):
        """ Test case where there are zero hits to ensure hit_rate() handles division by zero. """
        sdt = SignalDetection(0, 10, 5, 5)  # Hit Rate should be 0.0
        self.exp.add_condition(sdt, "Zero Hits")
        self.assertEqual(sdt.hit_rate(), 0.0)

    def test_zero_false_alarm_rate(self):
        """ Test case where there are zero false alarms to ensure false_alarm_rate() handles division by zero. """
        sdt = SignalDetection(5, 5, 0, 10)  # False Alarm Rate should be 0.0
        self.exp.add_condition(sdt, "Zero False Alarms")
        self.assertEqual(sdt.false_alarm_rate(), 0.0)

    def test_max_hit_rate(self):
        """ Test case where all trials are hits to ensure hit_rate() returns 1.0 """
        sdt = SignalDetection(10, 0, 5, 5)  # Hit Rate should be 1.0
        self.exp.add_condition(sdt, "Max Hits")
        self.assertEqual(sdt.hit_rate(), 1.0)

    def test_max_false_alarm_rate(self):
        """ Test case where all trials are false alarms to ensure false_alarm_rate() returns 1.0 """
        sdt = SignalDetection(5, 5, 10, 0)  # False Alarm Rate should be 1.0
        self.exp.add_condition(sdt, "Max False Alarms")
        self.assertEqual(sdt.false_alarm_rate(), 1.0)

if __name__ == '__main__':
    unittest.main()
