"""Merchant Risk Model — LightGBM-based merchant risk scorer with rule-based heuristic fallback."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import MERCHANT_FEATURE_SCHEMA
from models.base import BaseModel

# Heuristic weights for merchant rule-based scoring.
# Feature indices match MERCHANT_FEATURE_SCHEMA order.
# Weights calibrated for payment processor merchant risk (chargebacks, fraud rates, velocity).
_MERCHANT_HEURISTIC_WEIGHTS: dict[str, float] = {
    "transaction_count_1h": 0.06,
    "transaction_count_24h": 0.04,
    "transaction_volume_24h": 0.06,
    "unique_customers_24h": 0.04,
    "avg_transaction_amount_30d": 0.0,
    "std_transaction_amount_30d": 0.05,
    "chargeback_rate_90d": 0.20,
    "refund_rate_90d": 0.10,
    "account_age_days": 0.08,
    "fraud_transaction_rate": 0.18,
    "high_risk_customer_ratio": 0.10,
    "cross_border_ratio": 0.06,
    "velocity_spike_flag": 0.10,
    "mcc_risk_score": 0.08,
    "avg_customer_risk_score": 0.08,
}


class MerchantRiskModel(BaseModel):
    """LightGBM merchant risk model with SHAP explainability.

    When no trained model artifact is loaded, falls back to rule-based
    heuristic scoring using merchant risk domain rules — ensuring meaningful
    risk scores from day one.
    """

    def __init__(self, version: str = "v1.0.0"):
        super().__init__(
            model_name="pesaflow-merchant-risk",
            version=version,
            feature_schema=MERCHANT_FEATURE_SCHEMA,
        )
        self._explainer: Any = None

    @property
    def using_heuristic(self) -> bool:
        """True when scoring with rule-based heuristics instead of trained ML."""
        return not self._is_loaded

    def load(self, model_path: str) -> None:
        """Load LightGBM model from file. Also loads calibrator if present."""
        import lightgbm as lgb

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

        logger.info(f"Merchant risk model loaded: {self.version} from {model_path}")

    def load_from_sklearn(self, model: Any) -> None:
        """Load a pre-trained sklearn-compatible LightGBM model."""
        self._model = model
        self._is_loaded = True
        self._load_timestamp = time.time()
        logger.info(f"Merchant model loaded from sklearn object: {self.version}")

    def predict(self, features: list[float]) -> float:
        """Score a single merchant. Returns risk probability 0.0-1.0.

        Falls back to rule-based heuristic when no trained model is loaded.
        """
        if not self.validate_features(features):
            raise ValueError(
                f"Feature count mismatch: expected {self.feature_schema.feature_count}, got {len(features)}"
            )

        if not self._is_loaded:
            return self._heuristic_predict(features)

        arr = np.array([features])

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(arr)
            score = float(proba[0][1])
        else:
            score = float(self._model.predict(arr)[0])

        return max(0.0, min(1.0, score))

    def predict_batch(self, features_batch: list[list[float]]) -> list[float]:
        """Score a batch of merchants."""
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

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        feature_names = self.feature_schema.feature_names
        contributions = []
        for i, name in enumerate(feature_names):
            contributions.append(
                {
                    "feature": name,
                    "value": float(features[i]),
                    "impact": float(values[i]),
                }
            )

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:10]

    def get_feature_importance(self) -> dict[str, float]:
        """Get model-level feature importance."""
        if not self._is_loaded:
            return dict(_MERCHANT_HEURISTIC_WEIGHTS)

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
        """Rule-based merchant risk scoring using payment industry domain knowledge.

        Focuses on key merchant risk signals: chargeback rates, fraud rates,
        velocity anomalies, and customer base risk. Calibrated for payment
        processor standards (Visa/Mastercard monitoring thresholds).
        """
        score = 0.10  # Low base — most merchants are legitimate

        # Chargeback rate — strongest merchant risk signal
        # Visa threshold: 0.9% (standard), 1.8% (excessive)
        chargeback_rate = features[6]
        if chargeback_rate > 0.018:
            score += 0.25
        elif chargeback_rate > 0.009:
            score += 0.15
        elif chargeback_rate > 0.005:
            score += 0.08

        # Refund rate — high refund rates indicate potential friendly fraud
        refund_rate = features[7]
        if refund_rate > 0.15:
            score += 0.12
        elif refund_rate > 0.08:
            score += 0.06

        # Fraud transaction rate — direct risk signal
        fraud_rate = features[9]
        if fraud_rate > 0.05:
            score += 0.20
        elif fraud_rate > 0.02:
            score += 0.12
        elif fraud_rate > 0.01:
            score += 0.06

        # Account age — new merchants are higher risk
        account_age = features[8]
        if account_age < 30:
            score += 0.10
        elif account_age < 90:
            score += 0.05

        # Velocity spike — sudden activity change
        if features[12] > 0:  # velocity_spike_flag
            score += 0.10

        # Transaction volume anomaly
        txn_count_1h = features[0]
        txn_count_24h = features[1]
        if txn_count_1h > 100:
            score += 0.06
        if txn_count_24h > 500:
            score += 0.04

        # High-risk customer ratio
        high_risk_ratio = features[10]
        if high_risk_ratio > 0.25:
            score += 0.10
        elif high_risk_ratio > 0.15:
            score += 0.05

        # Cross-border ratio — higher fraud risk
        cross_border = features[11]
        if cross_border > 0.5:
            score += 0.06
        elif cross_border > 0.3:
            score += 0.03

        # MCC risk score (already 0.0-1.0)
        mcc_risk = features[13]
        score += mcc_risk * 0.08

        # Average customer risk score (0.0-1.0)
        avg_customer_risk = features[14]
        score += avg_customer_risk * 0.08

        # Transaction amount volatility
        std_amount = features[5]
        avg_amount = features[4] if features[4] > 0 else 100.0
        if avg_amount > 0 and std_amount / avg_amount > 3.0:
            score += 0.05

        return max(0.0, min(1.0, round(score, 4)))

    def _heuristic_shap(self, features: list[float]) -> list[dict]:
        """Generate pseudo-SHAP values based on heuristic rule activations."""
        impacts = self._compute_heuristic_impacts(features)
        feature_names = self.feature_schema.feature_names

        contributions = []
        for i, name in enumerate(feature_names):
            contributions.append(
                {
                    "feature": name,
                    "value": float(features[i]),
                    "impact": impacts.get(name, 0.0),
                }
            )

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:10]

    def _compute_heuristic_impacts(self, features: list[float]) -> dict[str, float]:
        """Compute per-feature impact for heuristic explanations."""
        impacts: dict[str, float] = {}

        # Chargeback rate
        cb = features[6]
        impacts["chargeback_rate_90d"] = 0.25 if cb > 0.018 else (0.15 if cb > 0.009 else (0.08 if cb > 0.005 else 0.0))

        # Refund rate
        rr = features[7]
        impacts["refund_rate_90d"] = 0.12 if rr > 0.15 else (0.06 if rr > 0.08 else 0.0)

        # Fraud rate
        fr = features[9]
        impacts["fraud_transaction_rate"] = 0.20 if fr > 0.05 else (0.12 if fr > 0.02 else (0.06 if fr > 0.01 else 0.0))

        # Account age (protective when old)
        age = features[8]
        impacts["account_age_days"] = 0.10 if age < 30 else (0.05 if age < 90 else (-0.02 if age > 365 else 0.0))

        # Velocity spike
        impacts["velocity_spike_flag"] = 0.10 if features[12] > 0 else 0.0

        # Transaction counts
        impacts["transaction_count_1h"] = 0.06 if features[0] > 100 else 0.0
        impacts["transaction_count_24h"] = 0.04 if features[1] > 500 else 0.0

        # Volume
        impacts["transaction_volume_24h"] = 0.0
        impacts["unique_customers_24h"] = 0.0

        # Amount stats
        impacts["avg_transaction_amount_30d"] = 0.0
        avg_amount = features[4] if features[4] > 0 else 100.0
        impacts["std_transaction_amount_30d"] = 0.05 if (avg_amount > 0 and features[5] / avg_amount > 3.0) else 0.0

        # Customer risk
        hr = features[10]
        impacts["high_risk_customer_ratio"] = 0.10 if hr > 0.25 else (0.05 if hr > 0.15 else 0.0)

        cb_ratio = features[11]
        impacts["cross_border_ratio"] = 0.06 if cb_ratio > 0.5 else (0.03 if cb_ratio > 0.3 else 0.0)

        impacts["mcc_risk_score"] = features[13] * 0.08
        impacts["avg_customer_risk_score"] = features[14] * 0.08

        return impacts
