"""Stacking Ensemble Trainer — Trains stacked ensemble (LightGBM + XGBoost + RF → Logistic Regression)."""

import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_engineering.feature_registry import (
    AML_FEATURE_SCHEMA,
    FRAUD_FEATURE_SCHEMA,
    MERCHANT_FEATURE_SCHEMA,
    FeatureSchema,
)

# Default hyperparameters per algorithm per domain
DOMAIN_CONFIGS = {
    "fraud": {
        "schema": FRAUD_FEATURE_SCHEMA,
        "target_col": "is_fraud",
        "lgb_params": {
            "objective": "binary",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 10,
            "random_state": 42,
            "verbose": -1,
        },
        "xgb_params": {
            "objective": "binary:logistic",
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 10,
            "random_state": 42,
            "use_label_encoder": False,
            "verbosity": 0,
        },
        "rf_params": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "aml": {
        "schema": AML_FEATURE_SCHEMA,
        "target_col": "is_suspicious",
        "lgb_params": {
            "objective": "binary",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.03,
            "n_estimators": 600,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "scale_pos_weight": 15,
            "random_state": 42,
            "verbose": -1,
        },
        "xgb_params": {
            "objective": "binary:logistic",
            "max_depth": 8,
            "learning_rate": 0.03,
            "n_estimators": 600,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "scale_pos_weight": 15,
            "random_state": 42,
            "use_label_encoder": False,
            "verbosity": 0,
        },
        "rf_params": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "merchant": {
        "schema": MERCHANT_FEATURE_SCHEMA,
        "target_col": "is_risky",
        "lgb_params": {
            "objective": "binary",
            "num_leaves": 48,
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "scale_pos_weight": 12,
            "random_state": 42,
            "verbose": -1,
        },
        "xgb_params": {
            "objective": "binary:logistic",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "scale_pos_weight": 12,
            "random_state": 42,
            "use_label_encoder": False,
            "verbosity": 0,
        },
        "rf_params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
}


class StackingEnsembleTrainer:
    """Trains a stacked ensemble per domain using StackingClassifier.

    Layer 0 (base learners): LightGBM, XGBoost, Random Forest
    Layer 1 (meta-learner): Logistic Regression on out-of-fold predictions
    """

    def __init__(
        self,
        domain: str,
        output_dir: str | None = None,
        mlflow_tracking_uri: str = "http://localhost:5000",
    ):
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}. Must be one of {list(DOMAIN_CONFIGS.keys())}")

        self.domain = domain
        self.config = DOMAIN_CONFIGS[domain]
        self.schema: FeatureSchema = self.config["schema"]
        self.target_col = self.config["target_col"]
        self.output_dir = output_dir or f"./model_artifacts/{domain}/ensemble"

        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(f"pesaflow-{domain}-ensemble")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")

    def train(
        self,
        data_path: str | None = None,
        df: pd.DataFrame | None = None,
        version: str = "v1.0.0",
        cv_folds: int = 5,
    ) -> dict:
        """Train stacked ensemble model.

        Args:
            data_path: Path to parquet training data
            df: DataFrame (alternative to data_path)
            version: Model version string
            cv_folds: Number of cross-validation folds for stacking

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training stacked ensemble for {self.domain}: {version}")

        if df is None:
            if data_path is None:
                raise ValueError("Either data_path or df must be provided")
            df = pd.read_parquet(data_path)

        X = df[self.schema.feature_names].values
        y = df[self.target_col].values

        logger.info(f"Dataset: {len(df)} samples, {y.sum()} positive ({y.mean()*100:.2f}%)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Build stacking classifier
        estimators = [
            ("lgb", lgb.LGBMClassifier(**self.config["lgb_params"])),
            ("xgb", xgb.XGBClassifier(**self.config["xgb_params"])),
            ("rf", RandomForestClassifier(**self.config["rf_params"])),
        ]

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            ),
            cv=cv_folds,
            passthrough=False,
            n_jobs=-1,
        )

        logger.info("Fitting stacking ensemble...")
        stacking_model.fit(X_train, y_train)

        # Evaluate
        metrics = self._evaluate(stacking_model, X_test, y_test)

        # Evaluate individual base learners for comparison
        base_metrics = self._evaluate_base_learners(stacking_model, X_test, y_test)
        metrics["base_learner_metrics"] = base_metrics

        logger.info(f"Ensemble ROC-AUC: {metrics['roc_auc']:.4f}")
        for name, bm in base_metrics.items():
            logger.info(f"  {name} ROC-AUC: {bm['roc_auc']:.4f}")

        # Save artifacts
        self._save_artifacts(stacking_model, metrics, version)

        # Log to MLflow
        self._log_to_mlflow(stacking_model, metrics, version)

        return metrics

    def _evaluate(self, model: StackingClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the stacking model."""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        return {
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
        }

    def _evaluate_base_learners(
        self, model: StackingClassifier, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, dict]:
        """Evaluate each base learner individually for comparison."""
        results = {}
        for name, estimator in model.named_estimators_.items():
            try:
                y_proba = estimator.predict_proba(X_test)[:, 1]
                y_pred = estimator.predict(X_test)
                results[name] = {
                    "roc_auc": float(roc_auc_score(y_test, y_proba)),
                    "precision": float(precision_score(y_test, y_pred)),
                    "recall": float(recall_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred)),
                }
            except Exception as e:
                logger.warning(f"Failed to evaluate base learner {name}: {e}")
        return results

    def _save_artifacts(self, model: StackingClassifier, metrics: dict, version: str) -> None:
        """Save stacking model and metadata."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save full stacking model
        model_path = os.path.join(self.output_dir, "stacking_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Stacking model saved: {model_path}")

        # Save individual base learners
        for name, estimator in model.named_estimators_.items():
            learner_path = os.path.join(self.output_dir, f"base_{name}.pkl")
            with open(learner_path, "wb") as f:
                pickle.dump(estimator, f)

        # Save meta-learner
        meta_path = os.path.join(self.output_dir, "meta_learner.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(model.final_estimator_, f)

        # Save metadata
        metadata = {
            "domain": self.domain,
            "version": version,
            "model_type": "stacking_ensemble",
            "base_learners": ["lgb", "xgb", "rf"],
            "meta_learner": "logistic_regression",
            "feature_names": self.schema.feature_names,
            "feature_schema_hash": self.schema.schema_hash,
            "metrics": {k: v for k, v in metrics.items() if k != "base_learner_metrics"},
            "base_learner_metrics": metrics.get("base_learner_metrics", {}),
        }
        meta_json_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _log_to_mlflow(self, model: StackingClassifier, metrics: dict, version: str) -> None:
        """Log ensemble training to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{self.domain}-ensemble-{version}"):
                flat_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
                mlflow.log_metrics(flat_metrics)
                mlflow.log_params({
                    "domain": self.domain,
                    "model_type": "stacking_ensemble",
                    "base_learners": "lgb,xgb,rf",
                    "meta_learner": "logistic_regression",
                })
                mlflow.set_tag("model_version", version)
                mlflow.set_tag("model_type", f"{self.domain}_ensemble")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
