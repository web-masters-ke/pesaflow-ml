"""Base model class for all Pesaflow ML models."""

import hashlib
import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import FeatureSchema


class BaseModel(ABC):
    """Abstract base class for fraud/AML ML models."""

    def __init__(self, model_name: str, version: str, feature_schema: FeatureSchema):
        self.model_name = model_name
        self.version = version
        self.feature_schema = feature_schema
        self._model: Any = None
        self._is_loaded = False
        self._load_timestamp: float | None = None
        self._artifact_path: str | None = None
        self._calibrator: Any = None  # CalibratedClassifierCV or calibration mapping

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model from artifact path."""
        ...

    @abstractmethod
    def predict(self, features: list[float]) -> float:
        """Score a single transaction. Returns probability 0.0-1.0."""
        ...

    @abstractmethod
    def predict_batch(self, features_batch: list[list[float]]) -> list[float]:
        """Score a batch of transactions."""
        ...

    @abstractmethod
    def get_shap_values(self, features: list[float]) -> list[dict]:
        """Compute SHAP feature importance for a single prediction."""
        ...

    def predict_with_confidence(
        self,
        features: list[float],
        feature_coverage: float = 1.0,
        maturity_confidence: float = 1.0,
    ) -> tuple[float, float]:
        """Score a transaction and return (score, confidence).

        Confidence is computed from:
        - data maturity level (passed in as maturity_confidence)
        - feature coverage for this specific request
        - model calibration error (if calibrator is loaded)

        Args:
            features: Feature vector
            feature_coverage: Fraction of features with non-default values (0.0-1.0)
            maturity_confidence: Confidence floor from data maturity level (0.0-1.0)

        Returns:
            Tuple of (calibrated_score, confidence)
        """
        raw_score = self.predict(features)

        # Apply calibration if available
        score = self._apply_calibration(raw_score)

        # Compute confidence as geometric mean of signals
        calibration_quality = self._get_calibration_quality()
        confidence = (feature_coverage * maturity_confidence * calibration_quality) ** (1 / 3)
        confidence = max(0.0, min(1.0, confidence))

        return score, confidence

    def predict_batch_with_confidence(
        self,
        features_batch: list[list[float]],
        feature_coverages: list[float] | None = None,
        maturity_confidence: float = 1.0,
    ) -> list[tuple[float, float]]:
        """Score a batch and return list of (score, confidence) tuples."""
        raw_scores = self.predict_batch(features_batch)
        calibration_quality = self._get_calibration_quality()

        if feature_coverages is None:
            feature_coverages = [1.0] * len(features_batch)

        results = []
        for raw_score, coverage in zip(raw_scores, feature_coverages):
            score = self._apply_calibration(raw_score)
            confidence = (coverage * maturity_confidence * calibration_quality) ** (1 / 3)
            confidence = max(0.0, min(1.0, confidence))
            results.append((score, confidence))

        return results

    def load_calibrator(self, calibrator_path: str) -> None:
        """Load calibration artifact from file."""
        try:
            path = Path(calibrator_path)
            if not path.exists():
                logger.warning(f"Calibrator not found: {calibrator_path}")
                return

            with open(path, "rb") as f:
                self._calibrator = pickle.load(f)

            logger.info(f"Calibrator loaded for {self.model_name} from {calibrator_path}")
        except Exception as e:
            logger.warning(f"Failed to load calibrator for {self.model_name}: {e}")

    def _apply_calibration(self, raw_score: float) -> float:
        """Map raw score through calibration function if available."""
        if self._calibrator is None:
            return raw_score

        try:
            arr = np.array([[raw_score]])
            if hasattr(self._calibrator, "predict_proba"):
                calibrated = self._calibrator.predict_proba(arr)[0][1]
            elif hasattr(self._calibrator, "transform"):
                calibrated = float(self._calibrator.transform(arr)[0])
            else:
                return raw_score
            return max(0.0, min(1.0, float(calibrated)))
        except Exception:
            return raw_score

    def _get_calibration_quality(self) -> float:
        """Get calibration quality score (1.0 if calibrator loaded, 0.7 if not)."""
        return 1.0 if self._calibrator is not None else 0.7

    def validate_features(self, features: list[float]) -> bool:
        """Validate feature vector matches schema."""
        return self.feature_schema.validate_features(features)

    def get_metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "feature_schema_version": self.feature_schema.version,
            "feature_schema_hash": self.feature_schema.schema_hash,
            "is_loaded": self._is_loaded,
            "load_timestamp": self._load_timestamp,
            "artifact_path": self._artifact_path,
            "calibrated": self._calibrator is not None,
        }

    def compute_feature_hash(self, features: list[float]) -> str:
        """Hash feature snapshot for audit."""
        payload = json.dumps(features)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]
