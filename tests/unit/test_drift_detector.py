"""Unit tests for Drift Detection."""

import numpy as np

from monitoring.drift_detector import DriftDetector


class TestDriftDetector:
    def setup_method(self):
        self.detector = DriftDetector(
            psi_threshold=0.2,
            accuracy_drop_threshold=0.05,
            fraud_rate_deviation_threshold=0.15,
        )

    def test_no_drift_similar_distributions(self):
        baseline = np.random.beta(2, 5, 10000)
        self.detector.set_baseline(baseline, accuracy=0.95, fraud_rate=0.05)

        current = np.random.beta(2, 5, 10000)
        results = self.detector.check_drift(current, current_accuracy=0.94, current_fraud_rate=0.05)

        assert results["retrain_recommended"] is False
        assert results["psi"] < 0.2

    def test_drift_detected_shifted_distribution(self):
        baseline = np.random.beta(2, 5, 10000)
        self.detector.set_baseline(baseline, accuracy=0.95, fraud_rate=0.05)

        # Very different distribution
        current = np.random.uniform(0, 1, 10000)
        results = self.detector.check_drift(current)

        assert results["psi"] > 0.0  # Should be measurably different

    def test_accuracy_drop_detected(self):
        baseline = np.random.beta(2, 5, 10000)
        self.detector.set_baseline(baseline, accuracy=0.95, fraud_rate=0.05)

        results = self.detector.check_drift(
            baseline,
            current_accuracy=0.85,  # 10% drop
            current_fraud_rate=0.05,
        )

        assert results["accuracy_alert"] is True
        assert results["retrain_recommended"] is True

    def test_fraud_rate_deviation_detected(self):
        baseline = np.random.beta(2, 5, 10000)
        self.detector.set_baseline(baseline, accuracy=0.95, fraud_rate=0.05)

        results = self.detector.check_drift(
            baseline,
            current_accuracy=0.95,
            current_fraud_rate=0.10,  # 100% deviation
        )

        assert results["fraud_rate_alert"] is True

    def test_no_baseline_returns_zero_psi(self):
        current = np.random.beta(2, 5, 10000)
        psi = self.detector.compute_psi(current)
        assert psi == 0.0
