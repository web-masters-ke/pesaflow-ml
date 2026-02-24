"""BentoML Serving Layer — Adaptive batching for fraud, AML, and merchant risk models."""

import os
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
from loguru import logger

from feature_engineering.feature_registry import (
    AML_FEATURE_SCHEMA,
    FRAUD_FEATURE_SCHEMA,
    MERCHANT_FEATURE_SCHEMA,
)

# === Model Runners with Adaptive Batching ===

MODEL_STORAGE = os.environ.get("MODEL_STORAGE_PATH", "./model_artifacts")


class FraudRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import lightgbm as lgb

        model_path = Path(MODEL_STORAGE) / "fraud" / "model.txt"
        if model_path.exists():
            self.model = lgb.Booster(model_file=str(model_path))
            self.loaded = True
            logger.info(f"BentoML fraud model loaded from {model_path}")
        else:
            self.model = None
            self.loaded = False
            logger.warning("Fraud model artifact not found for BentoML runner")

    @bentoml.Runnable.method(batchable=True, batch_dim=0, max_batch_size=64, max_latency_ms=50)
    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.loaded:
            return np.ones(features.shape[0])
        scores = self.model.predict(features)
        return np.clip(scores, 0.0, 1.0)


class AMLRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import xgboost as xgb

        model_path = Path(MODEL_STORAGE) / "aml" / "model.json"
        if model_path.exists():
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            self.loaded = True
            logger.info(f"BentoML AML model loaded from {model_path}")
        else:
            self.model = None
            self.loaded = False
            logger.warning("AML model artifact not found for BentoML runner")

    @bentoml.Runnable.method(batchable=True, batch_dim=0, max_batch_size=64, max_latency_ms=50)
    def predict(self, features: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        if not self.loaded:
            return np.ones(features.shape[0])
        dmatrix = xgb.DMatrix(features, feature_names=AML_FEATURE_SCHEMA.feature_names)
        scores = self.model.predict(dmatrix)
        return np.clip(scores, 0.0, 1.0)


class MerchantRiskRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import lightgbm as lgb

        model_path = Path(MODEL_STORAGE) / "merchant" / "model.txt"
        if model_path.exists():
            self.model = lgb.Booster(model_file=str(model_path))
            self.loaded = True
            logger.info(f"BentoML merchant model loaded from {model_path}")
        else:
            self.model = None
            self.loaded = False
            logger.warning("Merchant model artifact not found for BentoML runner")

    @bentoml.Runnable.method(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=50)
    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.loaded:
            return np.full(features.shape[0], 0.5)
        scores = self.model.predict(features)
        return np.clip(scores, 0.0, 1.0)


# === Runner Instances ===

fraud_runner = bentoml.Runner(FraudRunner, name="fraud_runner")
aml_runner = bentoml.Runner(AMLRunner, name="aml_runner")
merchant_runner = bentoml.Runner(MerchantRiskRunner, name="merchant_runner")

# === BentoML Service ===

svc = bentoml.Service(
    "pesaflow-ml",
    runners=[fraud_runner, aml_runner, merchant_runner],
)


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def score_fraud(input_data: dict) -> dict:
    """Score a transaction for fraud risk."""
    features = input_data.get("features", [])
    if len(features) != FRAUD_FEATURE_SCHEMA.feature_count:
        return {"error": f"Expected {FRAUD_FEATURE_SCHEMA.feature_count} features, got {len(features)}"}

    arr = np.array([features], dtype=np.float64)
    scores = await fraud_runner.predict.async_run(arr)
    score = float(scores[0])

    return {
        "risk_score": round(score, 4),
        "model": "fraud",
        "feature_count": len(features),
    }


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def score_aml(input_data: dict) -> dict:
    """Score a transaction for AML risk."""
    features = input_data.get("features", [])
    if len(features) != AML_FEATURE_SCHEMA.feature_count:
        return {"error": f"Expected {AML_FEATURE_SCHEMA.feature_count} features, got {len(features)}"}

    arr = np.array([features], dtype=np.float64)
    scores = await aml_runner.predict.async_run(arr)
    score = float(scores[0])

    return {
        "risk_score": round(score, 4),
        "model": "aml",
        "feature_count": len(features),
    }


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def score_merchant(input_data: dict) -> dict:
    """Score merchant risk."""
    features = input_data.get("features", [])
    if len(features) != MERCHANT_FEATURE_SCHEMA.feature_count:
        return {"error": f"Expected {MERCHANT_FEATURE_SCHEMA.feature_count} features, got {len(features)}"}

    arr = np.array([features], dtype=np.float64)
    scores = await merchant_runner.predict.async_run(arr)
    score = float(scores[0])

    return {
        "risk_score": round(score, 4),
        "model": "merchant",
        "feature_count": len(features),
    }


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def score_batch(input_data: dict) -> dict:
    """Batch scoring for any model type."""
    model_type = input_data.get("model", "fraud")
    batch = input_data.get("batch", [])

    if not batch:
        return {"error": "Empty batch"}

    arr = np.array(batch, dtype=np.float64)

    if model_type == "fraud":
        scores = await fraud_runner.predict.async_run(arr)
    elif model_type == "aml":
        scores = await aml_runner.predict.async_run(arr)
    elif model_type == "merchant":
        scores = await merchant_runner.predict.async_run(arr)
    else:
        return {"error": f"Unknown model type: {model_type}"}

    return {
        "model": model_type,
        "scores": [round(float(s), 4) for s in scores],
        "count": len(scores),
    }


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def health(_: dict) -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "runners": {
            "fraud": fraud_runner.init_local is not None,
            "aml": aml_runner.init_local is not None,
            "merchant": merchant_runner.init_local is not None,
        },
    }
