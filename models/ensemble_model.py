"""Ensemble Model — Serves stacked ensemble at inference time."""

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import (
    AML_FEATURE_SCHEMA,
    FRAUD_FEATURE_SCHEMA,
    MERCHANT_FEATURE_SCHEMA,
    FeatureSchema,
)
from models.base import BaseModel

DOMAIN_SCHEMAS = {
    "fraud": FRAUD_FEATURE_SCHEMA,
    "aml": AML_FEATURE_SCHEMA,
    "merchant": MERCHANT_FEATURE_SCHEMA,
}


class EnsembleModel(BaseModel):
    """Stacked ensemble model: loads and serves LightGBM + XGBoost + RF → LogReg meta-learner."""

    def __init__(self, domain: str, version: str = "v1.0.0"):
        schema = DOMAIN_SCHEMAS.get(domain)
        if not schema:
            raise ValueError(f"Unknown domain: {domain}")

        super().__init__(
            model_name=f"pesaflow-{domain}-ensemble",
            version=version,
            feature_schema=schema,
        )
        self.domain = domain
        self._base_learners: dict[str, Any] = {}
        self._meta_learner: Any = None
        self._executor = ThreadPoolExecutor(max_workers=3)

    def load(self, model_path: str) -> None:
        """Load all ensemble components from directory.

        Expected directory structure:
            model_path/
                stacking_model.pkl   (full StackingClassifier — preferred)
                base_lgb.pkl         (fallback: individual base learners)
                base_xgb.pkl
                base_rf.pkl
                meta_learner.pkl
                calibrator.pkl       (optional)
        """
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Ensemble directory not found: {model_path}")

        # Try loading full stacking model first
        stacking_path = path / "stacking_model.pkl"
        if stacking_path.exists():
            with open(stacking_path, "rb") as f:
                self._model = pickle.load(f)
            self._artifact_path = model_path
            self._is_loaded = True
            self._load_timestamp = time.time()

            # Extract components for SHAP
            if hasattr(self._model, "named_estimators_"):
                self._base_learners = dict(self._model.named_estimators_)
            if hasattr(self._model, "final_estimator_"):
                self._meta_learner = self._model.final_estimator_

            logger.info(f"Ensemble model loaded (full): {self.version} from {model_path}")
        else:
            # Load individual components
            for name in ("lgb", "xgb", "rf"):
                learner_path = path / f"base_{name}.pkl"
                if learner_path.exists():
                    with open(learner_path, "rb") as f:
                        self._base_learners[name] = pickle.load(f)

            meta_path = path / "meta_learner.pkl"
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    self._meta_learner = pickle.load(f)

            if self._base_learners and self._meta_learner:
                self._artifact_path = model_path
                self._is_loaded = True
                self._load_timestamp = time.time()
                logger.info(
                    f"Ensemble model loaded (components): {self.version}, "
                    f"base learners: {list(self._base_learners.keys())}"
                )
            else:
                raise RuntimeError("Could not load ensemble: missing base learners or meta-learner")

        # Load calibrator if present
        calibrator_path = path / "calibrator.pkl"
        if calibrator_path.exists():
            self.load_calibrator(str(calibrator_path))

    def predict(self, features: list[float]) -> float:
        """Score a single sample through the ensemble."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if not self.validate_features(features):
            raise ValueError(
                f"Feature count mismatch: expected {self.feature_schema.feature_count}, got {len(features)}"
            )

        arr = np.array([features])

        if self._model is not None and hasattr(self._model, "predict_proba"):
            # Use full StackingClassifier
            proba = self._model.predict_proba(arr)
            score = float(proba[0][1])
        else:
            # Manual stacking: run base learners → meta-learner
            score = self._manual_predict(arr)[0]

        return max(0.0, min(1.0, score))

    def predict_batch(self, features_batch: list[list[float]]) -> list[float]:
        """Score a batch through the ensemble, parallelizing base learner inference."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        arr = np.array(features_batch)

        if self._model is not None and hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(arr)
            return [max(0.0, min(1.0, float(p[1]))) for p in proba]

        # Manual stacking with parallelized base learners
        return self._manual_predict(arr)

    def _manual_predict(self, arr: np.ndarray) -> list[float]:
        """Run base learners in parallel, then feed to meta-learner."""

        def _predict_base(name_model: tuple[str, Any]) -> np.ndarray:
            _, model = name_model
            return model.predict_proba(arr)[:, 1]

        # Parallel base learner inference
        futures = list(self._executor.map(_predict_base, self._base_learners.items()))
        base_predictions = np.column_stack(futures)

        # Meta-learner
        meta_proba = self._meta_learner.predict_proba(base_predictions)
        return [max(0.0, min(1.0, float(p[1]))) for p in meta_proba]

    def get_shap_values(self, features: list[float]) -> list[dict]:
        """Compute SHAP values using the base learner with highest feature importance.

        Uses the LightGBM or XGBoost base learner since tree SHAP is most efficient.
        """
        import shap

        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        # Prefer LightGBM, fallback to XGBoost, then RF
        for name in ("lgb", "xgb", "rf"):
            if name in self._base_learners:
                model = self._base_learners[name]
                break
        else:
            return []

        arr = np.array([features])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(arr)

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

    def get_base_learner_names(self) -> list[str]:
        """Return names of loaded base learners."""
        return list(self._base_learners.keys())
