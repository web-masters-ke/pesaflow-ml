"""Merchant Risk Model — LightGBM-based merchant risk scorer."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import MERCHANT_FEATURE_SCHEMA
from models.base import BaseModel


class MerchantRiskModel(BaseModel):
    """LightGBM merchant risk model with SHAP explainability."""

    def __init__(self, version: str = "v1.0.0"):
        super().__init__(
            model_name="pesaflow-merchant-risk",
            version=version,
            feature_schema=MERCHANT_FEATURE_SCHEMA,
        )
        self._explainer: Any = None

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
        """Score a single merchant. Returns risk probability 0.0-1.0."""
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
            score = float(self._model.predict(arr)[0])

        return max(0.0, min(1.0, score))

    def predict_batch(self, features_batch: list[list[float]]) -> list[float]:
        """Score a batch of merchants."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        arr = np.array(features_batch)

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(arr)
            scores = [float(p[1]) for p in proba]
        else:
            raw_scores = self._model.predict(arr)
            scores = [max(0.0, min(1.0, float(s))) for s in raw_scores]

        return scores

    def get_shap_values(self, features: list[float]) -> list[dict]:
        """Compute SHAP values for explainability."""
        import shap

        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

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

        if hasattr(self._model, "feature_importance"):
            importance = self._model.feature_importance(importance_type="gain")
        elif hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        else:
            return {}

        feature_names = self.feature_schema.feature_names
        return {name: float(imp) for name, imp in zip(feature_names, importance)}
