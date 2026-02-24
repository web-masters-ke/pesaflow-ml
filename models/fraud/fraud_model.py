"""Fraud Detection Model — LightGBM-based binary classifier with rule-based heuristic fallback."""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import FRAUD_FEATURE_SCHEMA
from models.base import BaseModel

# Heuristic weights for rule-based scoring (used when no trained model exists).
# Feature indices match FRAUD_FEATURE_SCHEMA order.
# Weights represent domain-expert estimates of each feature's contribution to fraud risk.
_FRAUD_HEURISTIC_WEIGHTS: dict[str, dict] = {
    "avg_transaction_amount_7d": {"index": 0, "weight": 0.0},   # Used as baseline for ratio
    "transaction_velocity_1h":   {"index": 1, "weight": 0.15, "threshold": 5, "max_val": 20},
    "transaction_velocity_24h":  {"index": 2, "weight": 0.08, "threshold": 15, "max_val": 50},
    "failed_login_attempts_24h": {"index": 3, "weight": 0.10, "threshold": 2, "max_val": 10},
    "account_age_days":          {"index": 4, "weight": -0.10, "threshold": 30, "invert": True},
    "historical_fraud_flag":     {"index": 5, "weight": 0.20},
    "device_risk_score":         {"index": 6, "weight": 0.12},
    "device_fraud_count":        {"index": 7, "weight": 0.10, "threshold": 1, "max_val": 5},
    "distinct_user_count":       {"index": 8, "weight": 0.05, "threshold": 3, "max_val": 10},
    "amount":                    {"index": 9, "weight": 0.10},   # Ratio vs avg
    "geo_distance_from_last_tx": {"index": 10, "weight": 0.08, "threshold": 500, "max_val": 5000},
    "time_of_day":               {"index": 11, "weight": 0.04},  # Night hours
    "currency_risk":             {"index": 12, "weight": 0.08},
}


class FraudDetectionModel(BaseModel):
    """LightGBM fraud detection model with SHAP explainability.

    When no trained model artifact is loaded, falls back to rule-based
    heuristic scoring using domain-expert weights — ensuring meaningful
    risk scores from day one.
    """

    def __init__(self, version: str = "v1.0.0"):
        super().__init__(
            model_name="pesaflow-fraud-detector",
            version=version,
            feature_schema=FRAUD_FEATURE_SCHEMA,
        )
        self._explainer: Any = None

    @property
    def using_heuristic(self) -> bool:
        """True when scoring with rule-based heuristics instead of trained ML."""
        return not self._is_loaded

    def load(self, model_path: str) -> None:
        """Load LightGBM model from file. Also loads calibrator if present."""
        import lightgbm as lgb

        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found: {model_path}")

            self._model = lgb.Booster(model_file=str(path))
            self._artifact_path = model_path
            self._is_loaded = True
            self._load_timestamp = time.time()

            # Auto-load calibrator if it exists alongside the model
            calibrator_path = path.parent / "calibrator.pkl"
            if calibrator_path.exists():
                self.load_calibrator(str(calibrator_path))

            logger.info(f"Fraud model loaded: {self.version} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load fraud model: {e}")
            raise

    def load_from_sklearn(self, model: Any) -> None:
        """Load a pre-trained sklearn-compatible LightGBM model (for training pipeline)."""
        self._model = model
        self._is_loaded = True
        self._load_timestamp = time.time()
        logger.info(f"Fraud model loaded from sklearn object: {self.version}")

    def predict(self, features: list[float]) -> float:
        """Score a single transaction. Returns fraud probability 0.0-1.0.

        Falls back to rule-based heuristic when no trained model is loaded.
        """
        if not self.validate_features(features):
            raise ValueError(
                f"Feature count mismatch: expected {self.feature_schema.feature_count}, got {len(features)}"
            )

        if not self._is_loaded:
            return self._heuristic_predict(features)

        arr = np.array([features])

        # LightGBM Booster.predict returns probabilities directly
        if hasattr(self._model, "predict_proba"):
            # sklearn API
            proba = self._model.predict_proba(arr)
            score = float(proba[0][1])
        else:
            # Native Booster API
            score = float(self._model.predict(arr)[0])

        return max(0.0, min(1.0, score))

    def predict_batch(self, features_batch: list[list[float]]) -> list[float]:
        """Score a batch of transactions."""
        if not self._is_loaded:
            return [self._heuristic_predict(f) for f in features_batch]

        arr = np.array(features_batch)

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(arr)
            scores = [float(p[1]) for p in proba]
        else:
            raw_scores = self._model.predict(arr)
            scores = [max(0.0, min(1.0, float(s))) for s in raw_scores]

        return scores

    def get_shap_values(self, features: list[float]) -> list[dict]:
        """Compute SHAP values for explainability.

        When no trained model is loaded, returns heuristic-based pseudo-importance values.
        """
        if not self._is_loaded:
            return self._heuristic_shap(features)

        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)

        arr = np.array([features])
        shap_values = self._explainer.shap_values(arr)

        # For binary classification, shap_values may be a list of 2 arrays
        if isinstance(shap_values, list):
            values = shap_values[1][0]  # Positive class
        else:
            values = shap_values[0]

        feature_names = self.feature_schema.feature_names
        contributions = []
        for i, name in enumerate(feature_names):
            contributions.append({
                "feature": name,
                "value": float(features[i]),
                "impact": float(values[i]),
            })

        # Sort by absolute impact descending
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:10]  # Top 10

    def get_feature_importance(self) -> dict[str, float]:
        """Get model-level feature importance."""
        if not self._is_loaded:
            return {
                name: abs(cfg.get("weight", 0.0))
                for name, cfg in _FRAUD_HEURISTIC_WEIGHTS.items()
            }

        if hasattr(self._model, "feature_importance"):
            importance = self._model.feature_importance(importance_type="gain")
        elif hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        else:
            return {}

        feature_names = self.feature_schema.feature_names
        return {name: float(imp) for name, imp in zip(feature_names, importance)}

    # ------------------------------------------------------------------
    # Heuristic fallback methods
    # ------------------------------------------------------------------

    def _heuristic_predict(self, features: list[float]) -> float:
        """Rule-based fraud scoring using domain-expert weighted signals.

        Combines multiple risk signals with calibrated weights to produce
        a probability-like score in [0.0, 1.0]. Designed to be conservative
        (slightly over-predicts risk) to maintain safety before ML training.
        """
        score = 0.10  # Low base risk — most transactions are legitimate

        avg_7d = features[0] if features[0] > 0 else 100.0
        amount = features[9]

        # Amount anomaly: ratio of current amount to 7-day average
        if avg_7d > 0:
            amount_ratio = amount / avg_7d
            if amount_ratio > 5.0:
                score += 0.15
            elif amount_ratio > 3.0:
                score += 0.08
            elif amount_ratio > 2.0:
                score += 0.04

        # Transaction velocity (1h) — rapid-fire transactions
        vel_1h = features[1]
        if vel_1h > 10:
            score += 0.15
        elif vel_1h > 5:
            score += 0.08

        # Transaction velocity (24h)
        vel_24h = features[2]
        if vel_24h > 30:
            score += 0.08
        elif vel_24h > 15:
            score += 0.04

        # Failed login attempts
        failed_logins = features[3]
        if failed_logins > 5:
            score += 0.10
        elif failed_logins > 2:
            score += 0.05

        # New account (< 7 days) with high amount
        account_age = features[4]
        if account_age < 7 and amount > 500:
            score += 0.15
        elif account_age < 30 and amount > 1000:
            score += 0.08

        # Historical fraud flag — strongest signal
        if features[5] > 0:
            score += 0.20

        # Device risk score (already 0.0-1.0)
        device_risk = features[6]
        score += device_risk * 0.12

        # Device fraud count
        if features[7] > 0:
            score += min(features[7] / 5.0, 1.0) * 0.10

        # Distinct users on same device
        if features[8] > 3:
            score += 0.05

        # Geo distance from last transaction (km)
        geo_dist = features[10]
        if geo_dist > 2000:
            score += 0.08
        elif geo_dist > 500:
            score += 0.04

        # Time of day risk (late night: 0-5 AM encoded as 0.0-0.21)
        time_of_day = features[11]
        if time_of_day < 0.21 or time_of_day > 0.92:  # ~0-5 AM or ~10 PM-midnight
            score += 0.04

        # Currency risk (already 0.0-1.0)
        currency_risk = features[12]
        score += currency_risk * 0.08

        return max(0.0, min(1.0, round(score, 4)))

    def _heuristic_shap(self, features: list[float]) -> list[dict]:
        """Generate pseudo-SHAP values based on heuristic rule activations."""
        contributions = []
        feature_names = self.feature_schema.feature_names
        base_score = 0.10

        # Compute each feature's marginal contribution
        impacts = self._compute_heuristic_impacts(features)

        for i, name in enumerate(feature_names):
            contributions.append({
                "feature": name,
                "value": float(features[i]),
                "impact": impacts.get(name, 0.0),
            })

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:10]

    def _compute_heuristic_impacts(self, features: list[float]) -> dict[str, float]:
        """Compute per-feature impact for heuristic explanations."""
        impacts: dict[str, float] = {}
        avg_7d = features[0] if features[0] > 0 else 100.0
        amount = features[9]

        # Amount ratio impact
        amount_ratio = amount / avg_7d if avg_7d > 0 else 1.0
        if amount_ratio > 5.0:
            impacts["amount"] = 0.15
        elif amount_ratio > 3.0:
            impacts["amount"] = 0.08
        elif amount_ratio > 2.0:
            impacts["amount"] = 0.04
        else:
            impacts["amount"] = 0.0

        # Velocity 1h
        vel_1h = features[1]
        impacts["transaction_velocity_1h"] = 0.15 if vel_1h > 10 else (0.08 if vel_1h > 5 else 0.0)

        # Velocity 24h
        vel_24h = features[2]
        impacts["transaction_velocity_24h"] = 0.08 if vel_24h > 30 else (0.04 if vel_24h > 15 else 0.0)

        # Failed logins
        fl = features[3]
        impacts["failed_login_attempts_24h"] = 0.10 if fl > 5 else (0.05 if fl > 2 else 0.0)

        # Account age (protective factor — negative impact means reducing risk)
        age = features[4]
        if age < 7 and amount > 500:
            impacts["account_age_days"] = 0.15
        elif age < 30 and amount > 1000:
            impacts["account_age_days"] = 0.08
        else:
            impacts["account_age_days"] = -0.02 if age > 365 else 0.0

        # Historical fraud
        impacts["historical_fraud_flag"] = 0.20 if features[5] > 0 else 0.0

        # Device risk
        impacts["device_risk_score"] = features[6] * 0.12

        # Device fraud count
        impacts["device_fraud_count"] = min(features[7] / 5.0, 1.0) * 0.10

        # Distinct users
        impacts["distinct_user_count"] = 0.05 if features[8] > 3 else 0.0

        # Avg amount (baseline reference)
        impacts["avg_transaction_amount_7d"] = 0.0

        # Geo distance
        geo = features[10]
        impacts["geo_distance_from_last_tx"] = 0.08 if geo > 2000 else (0.04 if geo > 500 else 0.0)

        # Time of day
        tod = features[11]
        impacts["time_of_day"] = 0.04 if (tod < 0.21 or tod > 0.92) else 0.0

        # Currency risk
        impacts["currency_risk"] = features[12] * 0.08

        return impacts
