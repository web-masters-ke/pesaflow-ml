"""Isolation Forest Trainer — Unsupervised anomaly detection trained on ALL transactions (no labels needed)."""

import json
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from feature_engineering.feature_registry import (
    AML_FEATURE_SCHEMA,
    FRAUD_FEATURE_SCHEMA,
    MERCHANT_FEATURE_SCHEMA,
    FeatureSchema,
)

# Domain-specific contamination estimates
CONTAMINATION_RATES = {
    "fraud": 0.05,
    "aml": 0.03,
    "merchant": 0.04,
}

DOMAIN_SCHEMAS = {
    "fraud": FRAUD_FEATURE_SCHEMA,
    "aml": AML_FEATURE_SCHEMA,
    "merchant": MERCHANT_FEATURE_SCHEMA,
}

DOMAIN_TARGETS = {
    "fraud": "is_fraud",
    "aml": "is_suspicious",
    "merchant": "is_risky",
}


class IsolationForestTrainer:
    """Trains an Isolation Forest per domain for unsupervised anomaly detection.

    Works from COLD start — no labels required.
    Anomaly scores are blended with supervised model scores, weighted by maturity level.
    """

    def __init__(
        self,
        domain: str,
        output_dir: str | None = None,
        contamination: float | None = None,
        mlflow_tracking_uri: str = "http://localhost:5000",
    ):
        if domain not in DOMAIN_SCHEMAS:
            raise ValueError(f"Unknown domain: {domain}")

        self.domain = domain
        self.schema: FeatureSchema = DOMAIN_SCHEMAS[domain]
        self.target_col = DOMAIN_TARGETS[domain]
        self.contamination = contamination or CONTAMINATION_RATES.get(domain, 0.05)
        self.output_dir = output_dir or f"./model_artifacts/{domain}/anomaly"

        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(f"pesaflow-{domain}-anomaly")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")

    def train(
        self,
        data_path: str | None = None,
        df: pd.DataFrame | None = None,
        version: str = "v1.0.0",
        n_estimators: int = 200,
        max_samples: str | int = "auto",
    ) -> dict:
        """Train Isolation Forest.

        Args:
            data_path: Path to parquet data (uses ALL data, labels optional for evaluation)
            df: DataFrame (alternative to data_path)
            version: Model version string
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training Isolation Forest for {self.domain}: {version}")

        if df is None:
            if data_path is None:
                raise ValueError("Either data_path or df must be provided")
            df = pd.read_parquet(data_path)

        X = df[self.schema.feature_names].values
        has_labels = self.target_col in df.columns
        y = df[self.target_col].values if has_labels else None

        logger.info(f"Dataset: {len(df)} samples (labels available: {has_labels})")

        # Train Isolation Forest (unsupervised — uses all data)
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)

        # Get anomaly scores (higher = more anomalous, normalized to 0-1)
        raw_scores = model.decision_function(X)
        # Invert and normalize: decision_function returns lower for anomalies
        anomaly_scores = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)

        # Evaluate if labels are available
        metrics: dict = {
            "n_samples": len(X),
            "n_estimators": n_estimators,
            "contamination": self.contamination,
            "anomaly_score_mean": float(np.mean(anomaly_scores)),
            "anomaly_score_std": float(np.std(anomaly_scores)),
        }

        if has_labels and y is not None:
            # ROC-AUC: how well anomaly scores separate positive from negative
            roc = roc_auc_score(y, anomaly_scores)
            metrics["roc_auc_vs_labels"] = float(roc)
            logger.info(f"Isolation Forest ROC-AUC vs labels: {roc:.4f}")

            # Check anomaly detection rates
            predictions = model.predict(X)  # -1 = anomaly, 1 = normal
            flagged = predictions == -1
            if y.sum() > 0:
                recall = (flagged & (y == 1)).sum() / y.sum()
                metrics["anomaly_recall"] = float(recall)
                logger.info(f"Anomaly recall (% of positives flagged): {recall:.2%}")

        # Save artifacts
        self._save_artifacts(model, metrics, version)

        # Log to MLflow
        self._log_to_mlflow(model, metrics, version, n_estimators)

        logger.info(f"Isolation Forest training complete: {version}")
        return metrics

    def _save_artifacts(self, model: IsolationForest, metrics: dict, version: str) -> None:
        """Save model and metadata."""
        os.makedirs(self.output_dir, exist_ok=True)

        model_path = os.path.join(self.output_dir, "isolation_forest.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Isolation Forest saved: {model_path}")

        metadata = {
            "domain": self.domain,
            "version": version,
            "model_type": "isolation_forest",
            "feature_names": self.schema.feature_names,
            "feature_schema_hash": self.schema.schema_hash,
            "contamination": self.contamination,
            "metrics": metrics,
        }
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _log_to_mlflow(self, model: IsolationForest, metrics: dict, version: str, n_estimators: int) -> None:
        """Log to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{self.domain}-iforest-{version}"):
                mlflow.log_params(
                    {
                        "domain": self.domain,
                        "model_type": "isolation_forest",
                        "n_estimators": n_estimators,
                        "contamination": self.contamination,
                    }
                )
                safe_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(safe_metrics)
                mlflow.set_tag("model_version", version)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
