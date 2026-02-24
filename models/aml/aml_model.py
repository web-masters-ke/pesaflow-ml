"""AML Risk Scoring Model — XGBoost-based binary classifier."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import AML_FEATURE_SCHEMA
from models.base import BaseModel


class AMLRiskModel(BaseModel):
    """XGBoost AML risk scoring model with SHAP explainability."""

    def __init__(self, version: str = "v1.0.0"):
        super().__init__(
            model_name="pesaflow-aml-scorer",
            version=version,
            feature_schema=AML_FEATURE_SCHEMA,
        )
        self._explainer: Any = None

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
        """Score a single transaction for AML risk. Returns probability 0.0-1.0."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if not self.validate_features(features):
            raise ValueError(
                f"Feature count mismatch: expected {self.feature_schema.feature_count}, got {len(features)}"
            )

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
            raise RuntimeError("Model not loaded")

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
        """Compute SHAP values for AML explainability."""
        import shap

        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

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
        contributions = []
        for i, name in enumerate(feature_names):
            contributions.append({
                "feature": name,
                "value": float(features[i]),
                "impact": float(values[i]),
            })

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:10]

    def get_feature_importance(self) -> dict[str, float]:
        """Get model-level feature importance."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

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
