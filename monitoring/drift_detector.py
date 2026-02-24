"""Model Drift Detection — PSI, Wasserstein distance, per-feature drift, confidence drift, and label delay monitoring."""

import asyncio
import json
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import wasserstein_distance


class DriftDetector:
    """Monitors model drift using PSI, Wasserstein distance, per-feature analysis, and confidence tracking."""

    def __init__(
        self,
        psi_threshold: float = 0.2,
        accuracy_drop_threshold: float = 0.05,
        fraud_rate_deviation_threshold: float = 0.15,
        wasserstein_threshold: float = 0.1,
        confidence_drop_threshold: float = 0.1,
        label_delay_warning_hours: float = 72.0,
        db_pool: Any = None,
    ):
        self.psi_threshold = psi_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.fraud_rate_deviation = fraud_rate_deviation_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.label_delay_warning_hours = label_delay_warning_hours
        self._db = db_pool

        self._baseline_distribution: np.ndarray | None = None
        self._baseline_accuracy: float | None = None
        self._baseline_fraud_rate: float | None = None
        self._baseline_feature_distributions: dict[str, np.ndarray] | None = None
        self._baseline_confidence_mean: float | None = None

    def set_baseline(
        self,
        score_distribution: np.ndarray,
        accuracy: float,
        fraud_rate: float,
        feature_distributions: dict[str, np.ndarray] | None = None,
        confidence_mean: float | None = None,
    ) -> None:
        """Set baseline metrics from training or last stable period."""
        self._baseline_distribution = score_distribution
        self._baseline_accuracy = accuracy
        self._baseline_fraud_rate = fraud_rate
        self._baseline_feature_distributions = feature_distributions
        self._baseline_confidence_mean = confidence_mean
        logger.info(f"Drift baseline set: accuracy={accuracy:.4f}, fraud_rate={fraud_rate:.4f}")

    def compute_psi(self, current_distribution: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index between baseline and current score distribution."""
        if self._baseline_distribution is None:
            logger.warning("No baseline set, cannot compute PSI")
            return 0.0

        bins = np.linspace(0, 1, n_bins + 1)
        baseline_counts, _ = np.histogram(self._baseline_distribution, bins=bins)
        current_counts, _ = np.histogram(current_distribution, bins=bins)

        baseline_pct = (baseline_counts + 1) / (baseline_counts.sum() + n_bins)
        current_pct = (current_counts + 1) / (current_counts.sum() + n_bins)

        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return float(psi)

    def compute_feature_psi(
        self,
        current_features: dict[str, np.ndarray],
        n_bins: int = 10,
    ) -> dict[str, float]:
        """Compute PSI for each feature individually. Flags features that drift most."""
        if self._baseline_feature_distributions is None:
            return {}

        feature_psi = {}
        for name, current_dist in current_features.items():
            baseline_dist = self._baseline_feature_distributions.get(name)
            if baseline_dist is None:
                continue

            # Use adaptive binning based on combined data range
            combined = np.concatenate([baseline_dist, current_dist])
            bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

            baseline_counts, _ = np.histogram(baseline_dist, bins=bins)
            current_counts, _ = np.histogram(current_dist, bins=bins)

            baseline_pct = (baseline_counts + 1) / (baseline_counts.sum() + n_bins)
            current_pct = (current_counts + 1) / (current_counts.sum() + n_bins)

            psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
            feature_psi[name] = round(psi, 6)

        return feature_psi

    def compute_wasserstein(
        self,
        current_distribution: np.ndarray,
    ) -> float:
        """Compute Wasserstein distance between baseline and current distributions.

        Better than PSI for continuous distributions — measures the "earth mover's distance".
        """
        if self._baseline_distribution is None:
            return 0.0

        return float(wasserstein_distance(self._baseline_distribution, current_distribution))

    def compute_feature_wasserstein(
        self,
        current_features: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute Wasserstein distance per feature."""
        if self._baseline_feature_distributions is None:
            return {}

        distances = {}
        for name, current_dist in current_features.items():
            baseline_dist = self._baseline_feature_distributions.get(name)
            if baseline_dist is None:
                continue
            distances[name] = round(float(wasserstein_distance(baseline_dist, current_dist)), 6)

        return distances

    def check_confidence_drift(
        self,
        current_confidences: np.ndarray,
    ) -> dict:
        """Track if model confidence is systematically dropping."""
        current_mean = float(np.mean(current_confidences))
        current_std = float(np.std(current_confidences))

        result = {
            "current_confidence_mean": round(current_mean, 4),
            "current_confidence_std": round(current_std, 4),
            "confidence_drift_alert": False,
            "confidence_drop": None,
        }

        if self._baseline_confidence_mean is not None:
            drop = self._baseline_confidence_mean - current_mean
            result["confidence_drop"] = round(drop, 4)
            if drop > self.confidence_drop_threshold:
                result["confidence_drift_alert"] = True

        return result

    def check_label_delay(self, avg_label_delay_hours: float) -> dict:
        """Flag if the feedback loop (label confirmation) is too slow."""
        return {
            "avg_label_delay_hours": round(avg_label_delay_hours, 2),
            "label_delay_alert": avg_label_delay_hours > self.label_delay_warning_hours,
        }

    def check_drift(
        self,
        current_scores: np.ndarray,
        current_accuracy: float | None = None,
        current_fraud_rate: float | None = None,
        model_version: str = "unknown",
        current_features: dict[str, np.ndarray] | None = None,
        current_confidences: np.ndarray | None = None,
        avg_label_delay_hours: float | None = None,
    ) -> dict:
        """Run all drift checks and return comprehensive results."""
        results: dict = {
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat(),
            "psi": None,
            "psi_alert": False,
            "wasserstein_distance": None,
            "wasserstein_alert": False,
            "accuracy_drop": None,
            "accuracy_alert": False,
            "fraud_rate_deviation": None,
            "fraud_rate_alert": False,
            "confidence_drift": None,
            "label_delay": None,
            "feature_drift": {},
            "top_drifting_features": [],
            "retrain_recommended": False,
            "alerts": [],
        }

        # PSI check (score distribution)
        psi = self.compute_psi(current_scores)
        results["psi"] = round(psi, 4)
        if psi > self.psi_threshold:
            results["psi_alert"] = True
            results["alerts"].append(f"PSI={psi:.4f} exceeds threshold {self.psi_threshold}")

        # Wasserstein distance check
        wd = self.compute_wasserstein(current_scores)
        results["wasserstein_distance"] = round(wd, 4)
        if wd > self.wasserstein_threshold:
            results["wasserstein_alert"] = True
            results["alerts"].append(f"Wasserstein distance={wd:.4f} exceeds threshold {self.wasserstein_threshold}")

        # Accuracy check
        if current_accuracy is not None and self._baseline_accuracy is not None:
            drop = self._baseline_accuracy - current_accuracy
            results["accuracy_drop"] = round(drop, 4)
            if drop > self.accuracy_drop_threshold:
                results["accuracy_alert"] = True
                results["alerts"].append(f"Accuracy dropped by {drop:.4f}")

        # Fraud rate check
        if current_fraud_rate is not None and self._baseline_fraud_rate is not None:
            deviation = abs(current_fraud_rate - self._baseline_fraud_rate) / max(self._baseline_fraud_rate, 0.001)
            results["fraud_rate_deviation"] = round(deviation, 4)
            if deviation > self.fraud_rate_deviation:
                results["fraud_rate_alert"] = True
                results["alerts"].append(f"Fraud rate deviation: {deviation:.2%}")

        # Per-feature drift
        if current_features:
            feature_psi = self.compute_feature_psi(current_features)
            feature_wd = self.compute_feature_wasserstein(current_features)
            results["feature_drift"] = {
                "psi": feature_psi,
                "wasserstein": feature_wd,
            }
            # Top drifting features (by PSI)
            drifting = [(name, psi_val) for name, psi_val in feature_psi.items() if psi_val > self.psi_threshold]
            drifting.sort(key=lambda x: x[1], reverse=True)
            results["top_drifting_features"] = [
                {"feature": name, "psi": psi_val} for name, psi_val in drifting[:10]
            ]
            if drifting:
                results["alerts"].append(
                    f"{len(drifting)} features drifting: {', '.join(name for name, _ in drifting[:5])}"
                )

        # Confidence drift
        if current_confidences is not None:
            conf_result = self.check_confidence_drift(current_confidences)
            results["confidence_drift"] = conf_result
            if conf_result["confidence_drift_alert"]:
                results["alerts"].append(
                    f"Confidence dropping: {conf_result['confidence_drop']:.4f} below baseline"
                )

        # Label delay monitoring
        if avg_label_delay_hours is not None:
            delay_result = self.check_label_delay(avg_label_delay_hours)
            results["label_delay"] = delay_result
            if delay_result["label_delay_alert"]:
                results["alerts"].append(
                    f"Label delay {avg_label_delay_hours:.1f}h exceeds {self.label_delay_warning_hours}h threshold"
                )

        # Recommend retraining if any alert triggered
        results["retrain_recommended"] = any([
            results["psi_alert"],
            results["wasserstein_alert"],
            results["accuracy_alert"],
            results["fraud_rate_alert"],
        ])

        if results["retrain_recommended"]:
            logger.warning(f"Drift detected for {model_version}: {results['alerts']}")
        else:
            logger.info(f"No drift detected for {model_version}: PSI={psi:.4f}, WD={wd:.4f}")

        if self._db:
            self._store_drift_metrics(results)

        return results

    def _store_drift_metrics(self, results: dict) -> None:
        """Persist drift metrics to database with retry on transient failures."""
        if not self._db:
            return

        try:
            # Run the async store in the current event loop if available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._async_store_drift_metrics(results))
            else:
                loop.run_until_complete(self._async_store_drift_metrics(results))
        except RuntimeError:
            # No event loop available, log and skip
            logger.warning("No event loop available for storing drift metrics")

    async def _async_store_drift_metrics(self, results: dict, max_attempts: int = 3) -> None:
        """Async implementation of drift metrics persistence with retry."""
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._db.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO drift_metrics
                        (id, model_version, psi, wasserstein_distance, accuracy_drop,
                         fraud_rate_deviation, retrain_recommended, alerts, feature_drift, created_at)
                        VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        """,
                        results.get("model_version", "unknown"),
                        results.get("psi"),
                        results.get("wasserstein_distance"),
                        results.get("accuracy_drop"),
                        results.get("fraud_rate_deviation"),
                        results.get("retrain_recommended", False),
                        json.dumps(results.get("alerts", [])),
                        json.dumps(results.get("feature_drift", {})),
                    )
                logger.info(f"Drift metrics stored for model {results.get('model_version')}")
                return
            except (ConnectionError, TimeoutError, OSError) as e:
                if attempt < max_attempts:
                    logger.warning(f"Drift metrics store attempt {attempt} failed: {e}, retrying...")
                    await asyncio.sleep(0.1 * (2 ** (attempt - 1)))
                else:
                    logger.error(f"Failed to store drift metrics after {max_attempts} attempts: {e}")
            except Exception as e:
                logger.error(f"Failed to store drift metrics: {e}")
                return


class FraudDriftDetector(DriftDetector):
    """Fraud-specific drift detector."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class AMLDriftDetector(DriftDetector):
    """AML-specific drift detector."""

    def __init__(self, **kwargs: Any):
        super().__init__(
            psi_threshold=kwargs.get("psi_threshold", 0.2),
            accuracy_drop_threshold=kwargs.get("accuracy_drop_threshold", 0.05),
            fraud_rate_deviation_threshold=kwargs.get("fraud_rate_deviation_threshold", 0.15),
        )
