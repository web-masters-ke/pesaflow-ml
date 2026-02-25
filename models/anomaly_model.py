"""Anomaly Model — Serves Isolation Forest at inference time."""

import pickle
import time
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


class AnomalyModel(BaseModel):
    """Isolation Forest anomaly detection model.

    Returns anomaly scores normalized to 0.0-1.0 where higher = more anomalous.
    Designed to complement supervised models, especially during cold start.
    """

    def __init__(self, domain: str, version: str = "v1.0.0"):
        schema = DOMAIN_SCHEMAS.get(domain)
        if not schema:
            raise ValueError(f"Unknown domain: {domain}")

        super().__init__(
            model_name=f"pesaflow-{domain}-anomaly",
            version=version,
            feature_schema=schema,
        )
        self.domain = domain
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def load(self, model_path: str) -> None:
        """Load Isolation Forest from pickle file or directory.

        Accepts either:
            - Direct path to .pkl file
            - Directory containing isolation_forest.pkl
        """
        path = Path(model_path)

        if path.is_dir():
            pkl_path = path / "isolation_forest.pkl"
        else:
            pkl_path = path

        if not pkl_path.exists():
            raise FileNotFoundError(f"Anomaly model not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            self._model = pickle.load(f)

        self._artifact_path = str(pkl_path)
        self._is_loaded = True
        self._load_timestamp = time.time()

        # Compute score range for normalization (from offset + threshold)
        # We'll normalize dynamically per prediction
        logger.info(f"Anomaly model loaded: {self.version} from {pkl_path}")

    def predict(self, features: list[float]) -> float:
        """Score a single sample for anomalousness. Returns 0.0-1.0 (higher = more anomalous)."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if not self.validate_features(features):
            raise ValueError(
                f"Feature count mismatch: expected {self.feature_schema.feature_count}, got {len(features)}"
            )

        arr = np.array([features])
        raw_score = self._model.decision_function(arr)[0]

        # Normalize: decision_function returns lower values for anomalies
        # Use offset to center around 0, then sigmoid to map to 0-1
        normalized = self._sigmoid(-raw_score * 5)  # Scale factor for sensitivity
        return max(0.0, min(1.0, float(normalized)))

    def predict_batch(self, features_batch: list[list[float]]) -> list[float]:
        """Score a batch of samples."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        arr = np.array(features_batch)
        raw_scores = self._model.decision_function(arr)

        # Normalize batch
        normalized = self._sigmoid(-raw_scores * 5)
        return [max(0.0, min(1.0, float(s))) for s in normalized]

    def get_shap_values(self, features: list[float]) -> list[dict]:
        """Approximate feature importance for anomaly detection.

        Isolation Forest doesn't have native SHAP, so we use feature perturbation.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        arr = np.array([features])
        base_score = self._model.decision_function(arr)[0]

        feature_names = self.feature_schema.feature_names
        contributions = []

        for i, name in enumerate(feature_names):
            # Perturb feature to 0 and measure change
            perturbed = arr.copy()
            perturbed[0, i] = 0.0
            perturbed_score = self._model.decision_function(perturbed)[0]
            impact = base_score - perturbed_score  # Positive = feature contributes to anomaly

            contributions.append(
                {
                    "feature": name,
                    "value": float(features[i]),
                    "impact": float(impact),
                }
            )

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:10]

    def is_anomaly(self, features: list[float], threshold: float = 0.5) -> bool:
        """Check if a sample is anomalous above given threshold."""
        return self.predict(features) > threshold

    @staticmethod
    def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
        """Sigmoid function for normalizing scores."""
        return 1 / (1 + np.exp(-x))
