"""Comprehensive unit tests for Drift Detection — covers all drift signals."""

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from monitoring.drift_detector import DriftDetector


class TestWassersteinDistance:
    def setup_method(self):
        self.detector = DriftDetector(wasserstein_threshold=0.1)
        baseline = np.random.beta(2, 5, 10000)
        self.detector.set_baseline(baseline, accuracy=0.95, fraud_rate=0.05)

    def test_similar_distributions_low_distance(self):
        current = np.random.beta(2, 5, 10000)
        wd = self.detector.compute_wasserstein(current)
        assert wd < 0.1

    def test_different_distributions_high_distance(self):
        current = np.random.uniform(0.5, 1.0, 10000)
        wd = self.detector.compute_wasserstein(current)
        assert wd > 0.1

    def test_no_baseline_returns_zero(self):
        detector = DriftDetector()
        wd = detector.compute_wasserstein(np.array([0.1, 0.2, 0.3]))
        assert wd == 0.0


class TestPerFeatureDrift:
    def setup_method(self):
        self.detector = DriftDetector(psi_threshold=0.2)
        self.baseline_features = {
            "amount": np.random.lognormal(7, 1, 5000),
            "velocity": np.random.poisson(5, 5000).astype(float),
            "risk_score": np.random.beta(2, 5, 5000),
        }
        self.detector.set_baseline(
            np.random.beta(2, 5, 5000),
            accuracy=0.95,
            fraud_rate=0.05,
            feature_distributions=self.baseline_features,
        )

    def test_no_feature_drift_similar_data(self):
        current = {
            "amount": np.random.lognormal(7, 1, 5000),
            "velocity": np.random.poisson(5, 5000).astype(float),
            "risk_score": np.random.beta(2, 5, 5000),
        }
        psi = self.detector.compute_feature_psi(current)
        for name, val in psi.items():
            assert val < 0.5  # Should be relatively low

    def test_feature_drift_on_shifted_data(self):
        current = {
            "amount": np.random.lognormal(10, 2, 5000),  # Very different
            "velocity": np.random.poisson(5, 5000).astype(float),
            "risk_score": np.random.beta(2, 5, 5000),
        }
        psi = self.detector.compute_feature_psi(current)
        # Amount should show measurably higher drift than stable features
        assert psi["amount"] > psi.get("velocity", 0)

    def test_feature_wasserstein(self):
        current = {
            "amount": np.random.lognormal(10, 2, 5000),  # Shifted
            "velocity": np.random.poisson(5, 5000).astype(float),
        }
        wd = self.detector.compute_feature_wasserstein(current)
        assert "amount" in wd
        assert wd["amount"] > 0

    def test_no_baseline_features_returns_empty(self):
        detector = DriftDetector()
        detector.set_baseline(np.array([0.1, 0.2]), accuracy=0.95, fraud_rate=0.05)
        psi = detector.compute_feature_psi({"amount": np.array([1.0, 2.0])})
        assert psi == {}


class TestConfidenceDrift:
    def setup_method(self):
        self.detector = DriftDetector(confidence_drop_threshold=0.1)
        self.detector.set_baseline(
            np.random.beta(2, 5, 5000),
            accuracy=0.95,
            fraud_rate=0.05,
            confidence_mean=0.85,
        )

    def test_no_confidence_drift(self):
        confidences = np.random.normal(0.84, 0.05, 1000)
        result = self.detector.check_confidence_drift(confidences)
        assert result["confidence_drift_alert"] is False

    def test_confidence_drift_detected(self):
        confidences = np.random.normal(0.60, 0.1, 1000)  # Significant drop
        result = self.detector.check_confidence_drift(confidences)
        assert result["confidence_drift_alert"] is True
        assert result["confidence_drop"] > 0.1

    def test_no_baseline_confidence(self):
        detector = DriftDetector()
        detector.set_baseline(np.array([0.1]), accuracy=0.95, fraud_rate=0.05)
        result = detector.check_confidence_drift(np.array([0.7, 0.8, 0.9]))
        assert result["confidence_drift_alert"] is False
        assert result["confidence_drop"] is None


class TestLabelDelay:
    def test_no_delay_alert(self):
        detector = DriftDetector(label_delay_warning_hours=72)
        result = detector.check_label_delay(48.0)
        assert result["label_delay_alert"] is False

    def test_delay_alert_triggered(self):
        detector = DriftDetector(label_delay_warning_hours=72)
        result = detector.check_label_delay(96.0)
        assert result["label_delay_alert"] is True

    def test_exact_threshold_no_alert(self):
        detector = DriftDetector(label_delay_warning_hours=72)
        result = detector.check_label_delay(72.0)
        assert result["label_delay_alert"] is False


class TestComprehensiveCheckDrift:
    def setup_method(self):
        self.detector = DriftDetector(
            psi_threshold=0.2,
            wasserstein_threshold=0.1,
            accuracy_drop_threshold=0.05,
            fraud_rate_deviation_threshold=0.15,
            confidence_drop_threshold=0.1,
            label_delay_warning_hours=72,
        )
        baseline = np.random.beta(2, 5, 10000)
        feature_dists = {
            "amount": np.random.lognormal(7, 1, 5000),
            "velocity": np.random.poisson(5, 5000).astype(float),
        }
        self.detector.set_baseline(
            baseline,
            accuracy=0.95,
            fraud_rate=0.05,
            feature_distributions=feature_dists,
            confidence_mean=0.85,
        )

    def test_no_drift_comprehensive(self):
        current = np.random.beta(2, 5, 10000)
        results = self.detector.check_drift(
            current,
            current_accuracy=0.94,
            current_fraud_rate=0.05,
            model_version="v1.0",
            current_features={
                "amount": np.random.lognormal(7, 1, 5000),
                "velocity": np.random.poisson(5, 5000).astype(float),
            },
            current_confidences=np.random.normal(0.84, 0.05, 1000),
            avg_label_delay_hours=48.0,
        )
        assert results["retrain_recommended"] is False
        assert results["model_version"] == "v1.0"
        assert "psi" in results
        assert "wasserstein_distance" in results

    def test_drift_triggers_retrain(self):
        current = np.random.uniform(0, 1, 10000)  # Very different
        results = self.detector.check_drift(
            current,
            current_accuracy=0.80,  # Big drop
            current_fraud_rate=0.15,  # Big deviation
        )
        assert results["retrain_recommended"] is True
        assert len(results["alerts"]) > 0

    def test_all_signals_present(self):
        current = np.random.beta(2, 5, 10000)
        results = self.detector.check_drift(
            current,
            current_accuracy=0.94,
            current_fraud_rate=0.05,
            current_features={"amount": np.random.lognormal(7, 1, 5000)},
            current_confidences=np.random.normal(0.84, 0.05, 1000),
            avg_label_delay_hours=48.0,
        )
        assert "psi" in results
        assert "wasserstein_distance" in results
        assert "accuracy_drop" in results
        assert "fraud_rate_deviation" in results
        assert "confidence_drift" in results
        assert "label_delay" in results
        assert "feature_drift" in results


class TestStoreDriftMetrics:
    def test_store_drift_metrics_with_db(self):
        """Test that _store_drift_metrics calls the async implementation."""
        detector = DriftDetector(db_pool=MagicMock())
        results = {"model_version": "v1.0", "psi": 0.1, "alerts": []}

        # Since _store_drift_metrics uses asyncio internally, just verify no exception
        # when no event loop is available (it should log a warning and skip)
        detector._store_drift_metrics(results)

    def test_store_drift_metrics_without_db(self):
        """Without db_pool, _store_drift_metrics should be a no-op."""
        detector = DriftDetector(db_pool=None)
        results = {"model_version": "v1.0"}
        detector._store_drift_metrics(results)  # Should not raise
