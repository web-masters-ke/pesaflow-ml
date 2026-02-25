"""AML Risk Scoring Model — XGBoost-based binary classifier with rule-based heuristic fallback."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import AML_FEATURE_SCHEMA
from models.base import BaseModel

# Heuristic weights for AML rule-based scoring.
# Feature indices match AML_FEATURE_SCHEMA order.
# Weights calibrated for mobile money AML patterns (structuring, rapid drain, sanctions proximity).
_AML_HEURISTIC_WEIGHTS: dict[str, float] = {
    "amount": 0.05,
    "velocity_1h": 0.10,
    "velocity_24h": 0.06,
    "total_volume_24h": 0.08,
    "avg_amount_30d": 0.0,
    "std_amount_30d": 0.04,
    "time_of_day": 0.02,
    "is_cross_border": 0.08,
    "account_age_days": 0.05,
    "device_count_30d": 0.04,
    "ip_count_30d": 0.04,
    "new_device_flag": 0.06,
    "kyc_completeness_score": 0.08,
    "network_risk_score": 0.10,
    "circular_transfer_flag": 0.15,
    "shared_device_cluster_size": 0.06,
    "high_risk_country_flag": 0.15,
    "sanctions_proximity_score": 0.18,
    "ip_country_mismatch": 0.08,
    "historical_structuring_flag": 0.12,
    "structuring_score_24h": 0.14,
    "rapid_drain_flag": 0.12,
}


class AMLRiskModel(BaseModel):
    """XGBoost AML risk scoring model with SHAP explainability.

    When no trained model artifact is loaded, falls back to rule-based
    heuristic scoring using AML-specific domain rules — ensuring meaningful
    risk scores from day one.
    """

    def __init__(self, version: str = "v1.0.0"):
        super().__init__(
            model_name="pesaflow-aml-scorer",
            version=version,
            feature_schema=AML_FEATURE_SCHEMA,
        )
        self._explainer: Any = None

    @property
    def using_heuristic(self) -> bool:
        """True when scoring with rule-based heuristics instead of trained ML."""
        return not self._is_loaded

    def load(self, model_path: str) -> None:
        """Load XGBoost model from file. Also loads calibrator if present."""
        import xgboost as xgb

        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found: {model_path}")

            self._model = xgb.Booster()
            self._model.load_model(str(path))
            self._artifact_path = model_path
            self._is_loaded = True
            self._load_timestamp = time.time()

            # Auto-load calibrator if it exists alongside the model
            calibrator_path = path.parent / "calibrator.pkl"
            if calibrator_path.exists():
                self.load_calibrator(str(calibrator_path))

            logger.info(f"AML model loaded: {self.version} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load AML model: {e}")
            raise

    def load_from_sklearn(self, model: Any) -> None:
        """Load a pre-trained sklearn-compatible XGBoost model."""
        self._model = model
        self._is_loaded = True
        self._load_timestamp = time.time()
        logger.info(f"AML model loaded from sklearn object: {self.version}")

    def predict(self, features: list[float]) -> float:
        """Score a single transaction for AML risk. Returns probability 0.0-1.0.

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
            import xgboost as xgb

            dmatrix = xgb.DMatrix(arr, feature_names=self.feature_schema.feature_names)
            score = float(self._model.predict(dmatrix)[0])

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
            import xgboost as xgb

            dmatrix = xgb.DMatrix(arr, feature_names=self.feature_schema.feature_names)
            raw_scores = self._model.predict(dmatrix)
            scores = [max(0.0, min(1.0, float(s))) for s in raw_scores]

        return scores

    def get_shap_values(self, features: list[float]) -> list[dict]:
        """Compute SHAP values for AML explainability.

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
        elif len(shap_values.shape) == 3:
            values = shap_values[0, :, 1]
        else:
            values = shap_values[0]

        feature_names = self.feature_schema.feature_names
        contributions: list[dict[str, Any]] = []
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
            return dict(_AML_HEURISTIC_WEIGHTS)

        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif hasattr(self._model, "get_score"):
            score_dict = self._model.get_score(importance_type="gain")
            feature_names = self.feature_schema.feature_names
            return {name: score_dict.get(name, 0.0) for name in feature_names}
        else:
            return {}

        feature_names = self.feature_schema.feature_names
        return {name: float(imp) for name, imp in zip(feature_names, importance)}

    # ------------------------------------------------------------------
    # Heuristic fallback methods
    # ------------------------------------------------------------------

    def _heuristic_predict(self, features: list[float]) -> float:
        """Rule-based AML scoring using regulatory domain knowledge.

        Focuses on key AML red flags: structuring, sanctions proximity,
        high-risk jurisdictions, circular transfers, and rapid account drain.
        Calibrated for mobile money patterns common in East Africa.
        """
        score = 0.08  # Low base — most transactions are legitimate

        amount = features[0]
        avg_30d = features[4] if features[4] > 0 else 200.0

        # Amount anomaly vs 30-day average
        if avg_30d > 0:
            ratio = amount / avg_30d
            if ratio > 10.0:
                score += 0.10
            elif ratio > 5.0:
                score += 0.05

        # Velocity spike (1h)
        vel_1h = features[1]
        if vel_1h > 15:
            score += 0.12
        elif vel_1h > 8:
            score += 0.06

        # High total volume in 24h (potential structuring/layering)
        total_vol_24h = features[3]
        if total_vol_24h > 50000:
            score += 0.10
        elif total_vol_24h > 20000:
            score += 0.05

        # Cross-border transfer
        if features[7] > 0:  # is_cross_border
            score += 0.08

        # New account with significant activity
        account_age = features[8]
        if account_age < 14 and total_vol_24h > 5000:
            score += 0.10
        elif account_age < 30:
            score += 0.03

        # Multiple devices/IPs (potential account takeover or money mule network)
        if features[9] > 5:  # device_count_30d
            score += 0.05
        if features[10] > 10:  # ip_count_30d
            score += 0.05

        # New device flag
        if features[11] > 0:
            score += 0.04

        # KYC incompleteness (inverted — lower KYC = higher risk)
        kyc = features[12]
        if kyc < 0.3:
            score += 0.10
        elif kyc < 0.6:
            score += 0.05

        # Network risk score (already 0.0-1.0)
        network_risk = features[13]
        score += network_risk * 0.10

        # Circular transfer flag — strong AML signal
        if features[14] > 0:
            score += 0.15

        # Shared device cluster
        cluster_size = features[15]
        if cluster_size > 5:
            score += 0.06

        # High-risk country flag — regulatory requirement
        if features[16] > 0:
            score += 0.15

        # Sanctions proximity score (0.0-1.0)
        sanctions = features[17]
        if sanctions > 0.7:
            score += 0.20
        elif sanctions > 0.3:
            score += 0.10

        # IP-country mismatch
        if features[18] > 0:
            score += 0.06

        # Historical structuring flag
        if features[19] > 0:
            score += 0.12

        # Current structuring score (0.0-1.0)
        structuring = features[20]
        if structuring > 0.7:
            score += 0.15
        elif structuring > 0.4:
            score += 0.08

        # Rapid drain flag — account emptying pattern
        if features[21] > 0:
            score += 0.12

        return max(0.0, min(1.0, round(score, 4)))

    def _heuristic_shap(self, features: list[float]) -> list[dict]:
        """Generate pseudo-SHAP values based on heuristic rule activations."""
        impacts = self._compute_heuristic_impacts(features)
        feature_names = self.feature_schema.feature_names

        contributions: list[dict[str, Any]] = []
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
        amount = features[0]
        avg_30d = features[4] if features[4] > 0 else 200.0

        ratio = amount / avg_30d if avg_30d > 0 else 1.0
        impacts["amount"] = 0.10 if ratio > 10 else (0.05 if ratio > 5 else 0.0)
        impacts["avg_amount_30d"] = 0.0
        impacts["std_amount_30d"] = 0.0

        vel_1h = features[1]
        impacts["velocity_1h"] = 0.12 if vel_1h > 15 else (0.06 if vel_1h > 8 else 0.0)
        impacts["velocity_24h"] = 0.0

        total_vol = features[3]
        impacts["total_volume_24h"] = 0.10 if total_vol > 50000 else (0.05 if total_vol > 20000 else 0.0)

        impacts["time_of_day"] = 0.0
        impacts["is_cross_border"] = 0.08 if features[7] > 0 else 0.0

        age = features[8]
        impacts["account_age_days"] = (
            0.10 if (age < 14 and total_vol > 5000) else (0.03 if age < 30 else -0.02 if age > 365 else 0.0)
        )

        impacts["device_count_30d"] = 0.05 if features[9] > 5 else 0.0
        impacts["ip_count_30d"] = 0.05 if features[10] > 10 else 0.0
        impacts["new_device_flag"] = 0.04 if features[11] > 0 else 0.0

        kyc = features[12]
        impacts["kyc_completeness_score"] = -0.10 if kyc < 0.3 else (-0.05 if kyc < 0.6 else 0.0)

        impacts["network_risk_score"] = features[13] * 0.10
        impacts["circular_transfer_flag"] = 0.15 if features[14] > 0 else 0.0
        impacts["shared_device_cluster_size"] = 0.06 if features[15] > 5 else 0.0
        impacts["high_risk_country_flag"] = 0.15 if features[16] > 0 else 0.0

        sanctions = features[17]
        impacts["sanctions_proximity_score"] = 0.20 if sanctions > 0.7 else (0.10 if sanctions > 0.3 else 0.0)

        impacts["ip_country_mismatch"] = 0.06 if features[18] > 0 else 0.0
        impacts["historical_structuring_flag"] = 0.12 if features[19] > 0 else 0.0

        structuring = features[20]
        impacts["structuring_score_24h"] = 0.15 if structuring > 0.7 else (0.08 if structuring > 0.4 else 0.0)

        impacts["rapid_drain_flag"] = 0.12 if features[21] > 0 else 0.0

        return impacts
