"""Fraud Model Training Pipeline — LightGBM with k-fold CV, SHAP, MLflow, SMOTE, calibration, and Optuna."""

import hashlib
import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_engineering.feature_registry import FRAUD_FEATURE_SCHEMA


class FraudModelTrainer:
    """LightGBM fraud model training pipeline with SMOTE, calibration, and Optuna integration."""

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "pesaflow-fraud-detection",
    ):
        self.feature_names = FRAUD_FEATURE_SCHEMA.feature_names
        self.target_col = "is_fraud"
        self.experiment_name = experiment_name

        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"MLflow not available, training locally: {e}")

        self.params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
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
        }

    def train(
        self,
        data_path: str,
        output_dir: str = "./model_artifacts/fraud",
        version: str = "v1.0.0",
        oversampling_strategy: str | None = None,
        run_optuna: bool = False,
        optuna_n_trials: int = 100,
        calibrate: bool = True,
    ) -> dict:
        """Full training pipeline: load data → oversample → train → calibrate → evaluate → save.

        Args:
            data_path: Path to training data (parquet)
            output_dir: Where to save model artifacts
            version: Model version string
            oversampling_strategy: None, "smote", "adasyn", or "borderline_smote"
            run_optuna: Whether to run Optuna hyperparameter optimization first
            optuna_n_trials: Number of Optuna trials
            calibrate: Whether to apply probability calibration
        """
        logger.info(f"Starting fraud model training: {version}")

        # Load and prepare data
        df = pd.read_parquet(data_path)
        dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]

        X = df[self.feature_names].values
        y = df[self.target_col].values

        logger.info(f"Dataset: {len(df)} samples, {y.sum()} fraud ({y.mean()*100:.2f}%)")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Optuna hyperparameter optimization
        if run_optuna:
            self.params = self._run_optuna(X_train, y_train, optuna_n_trials)

        # Apply oversampling to training set only
        X_train_resampled, y_train_resampled = self._apply_oversampling(
            X_train, y_train, strategy=oversampling_strategy
        )

        # K-fold cross validation (on original training data)
        cv_scores = self._cross_validate(X_train, y_train)

        # Train final model (on resampled data)
        model = lgb.LGBMClassifier(**self.params)
        model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        # Calibrate probabilities
        calibrator = None
        if calibrate:
            calibrator = self._calibrate_model(model, X_test, y_test)

        # Evaluate
        metrics = self._evaluate(model, X_test, y_test, cv_scores, calibrator)

        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.txt")
        model.booster_.save_model(model_path)

        # Save calibrator
        if calibrator is not None:
            calibrator_path = os.path.join(output_dir, "calibrator.pkl")
            with open(calibrator_path, "wb") as f:
                pickle.dump(calibrator, f)
            logger.info(f"Calibrator saved: {calibrator_path}")

        # Save metadata
        metadata = {
            "version": version,
            "dataset_hash": dataset_hash,
            "feature_schema_version": FRAUD_FEATURE_SCHEMA.version,
            "feature_schema_hash": FRAUD_FEATURE_SCHEMA.schema_hash,
            "hyperparameters": self.params,
            "metrics": metrics,
            "feature_names": self.feature_names,
            "oversampling_strategy": oversampling_strategy,
            "calibrated": calibrate,
            "optuna_tuned": run_optuna,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Log to MLflow
        self._log_to_mlflow(model, metrics, metadata, version)

        logger.info(f"Fraud model training complete: {version}")
        logger.info(
            f"ROC-AUC: {metrics['roc_auc']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
        )

        return metrics

    def _apply_oversampling(
        self, X: np.ndarray, y: np.ndarray, strategy: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply oversampling to training data only.

        Args:
            X: Feature matrix
            y: Labels
            strategy: None, "smote", "adasyn", or "borderline_smote"

        Returns:
            Resampled (X, y) tuple
        """
        if strategy is None:
            return X, y

        try:
            from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

            samplers = {
                "smote": SMOTE(random_state=42),
                "adasyn": ADASYN(random_state=42),
                "borderline_smote": BorderlineSMOTE(random_state=42),
            }

            sampler = samplers.get(strategy)
            if sampler is None:
                logger.warning(f"Unknown oversampling strategy: {strategy}, skipping")
                return X, y

            X_res, y_res = sampler.fit_resample(X, y)
            logger.info(
                f"Oversampling ({strategy}): {len(X)} → {len(X_res)} samples " f"(positive: {y.sum()} → {y_res.sum()})"
            )
            return X_res, y_res
        except ImportError:
            logger.warning("imbalanced-learn not installed, skipping oversampling")
            return X, y

    def _calibrate_model(
        self, model: lgb.LGBMClassifier, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> CalibratedClassifierCV:
        """Apply isotonic regression calibration."""
        logger.info("Calibrating model probabilities (isotonic regression)")
        calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrator.fit(X_cal, y_cal)
        return calibrator

    def _run_optuna(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> dict:
        """Run Optuna hyperparameter optimization."""
        from training.optimization.hyperopt import OptunaTuner

        tuner = OptunaTuner(domain="fraud", algorithm="lightgbm", n_trials=n_trials)
        best_params = tuner.optimize(X, y)
        tuner.save_best_params()
        return best_params

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> list[float]:
        """K-fold stratified cross validation."""
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**self.params)
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )

            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, y_pred_proba)
            scores.append(auc)
            logger.info(f"Fold {fold+1}: AUC = {auc:.4f}")

        logger.info(f"CV AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return scores

    def _evaluate(
        self,
        model: lgb.LGBMClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_scores: list[float],
        calibrator: CalibratedClassifierCV | None = None,
    ) -> dict:
        """Evaluate model on test set, using calibrated scores if available."""
        if calibrator is not None:
            y_pred_proba = calibrator.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "cv_auc_mean": float(np.mean(cv_scores)),
            "cv_auc_std": float(np.std(cv_scores)),
            "calibrated": calibrator is not None,
        }

        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")
        return metrics

    def _log_to_mlflow(self, model: lgb.LGBMClassifier, metrics: dict, metadata: dict, version: str) -> None:
        """Log experiment to MLflow."""
        try:
            with mlflow.start_run(run_name=f"fraud-{version}"):
                safe_params = {k: str(v) for k, v in self.params.items()}
                mlflow.log_params(safe_params)
                safe_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(safe_metrics)
                mlflow.lightgbm.log_model(model, "model")
                mlflow.log_dict(metadata, "metadata.json")
                mlflow.set_tag("model_version", version)
                mlflow.set_tag("model_type", "fraud_detection")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


def run_training(data_path: str | None = None, version: str = "v1.0.0") -> dict:
    """Entry point for fraud model training."""
    if data_path is None:
        # Generate synthetic data
        from training.data_generator import generate_fraud_dataset

        os.makedirs("./training/data", exist_ok=True)
        data_path = "./training/data/fraud_training_data.parquet"
        df = generate_fraud_dataset(n_samples=50000)
        df.to_parquet(data_path, index=False)

    trainer = FraudModelTrainer()
    return trainer.train(data_path=data_path, version=version)


if __name__ == "__main__":
    run_training()
