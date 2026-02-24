"""Merchant Risk Model Training Pipeline — LightGBM with stratified CV, MLflow, SMOTE, calibration, and Optuna."""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_engineering.feature_registry import MERCHANT_FEATURE_SCHEMA
from models.merchant.merchant_model import MerchantRiskModel


class MerchantModelTrainer:
    """LightGBM-based merchant risk model trainer with SMOTE, calibration, and Optuna."""

    def __init__(self, output_dir: str = "./model_artifacts/merchant"):
        self.output_dir = output_dir
        self.feature_names = MERCHANT_FEATURE_SCHEMA.feature_names
        self.model_version = f"v{datetime.utcnow().strftime('%Y%m%d')}"

        self.hyperparams = {
            "objective": "binary",
            "metric": "auc",
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
            "verbosity": -1,
        }

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "is_risky",
        oversampling_strategy: str | None = None,
        run_optuna: bool = False,
        optuna_n_trials: int = 100,
        calibrate: bool = True,
    ) -> dict:
        """Full training pipeline with SMOTE, calibration, and Optuna.

        Args:
            df: Training DataFrame
            target_col: Name of the target column
            oversampling_strategy: None, "smote", "adasyn", or "borderline_smote"
            run_optuna: Whether to run Optuna hyperparameter optimization
            optuna_n_trials: Number of Optuna trials
            calibrate: Whether to apply probability calibration
        """
        logger.info(f"Training merchant model on {len(df)} samples")

        X = df[self.feature_names].values
        y = df[target_col].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Optuna hyperparameter optimization
        if run_optuna:
            self.hyperparams = self._run_optuna(X_train, y_train, optuna_n_trials)

        # Apply oversampling to training set only
        X_train_resampled, y_train_resampled = self._apply_oversampling(
            X_train, y_train, strategy=oversampling_strategy
        )

        # Cross-validation (on original data)
        cv_scores = self._cross_validate(X_train, y_train)
        logger.info(f"CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final model (on resampled data)
        model = lgb.LGBMClassifier(**self.hyperparams)
        model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        # Calibrate probabilities
        calibrator = None
        if calibrate:
            calibrator = self._calibrate_model(model, X_test, y_test)

        # Evaluate
        metrics = self._evaluate(model, X_test, y_test, calibrator)
        metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        metrics["cv_roc_auc_std"] = float(np.std(cv_scores))

        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}, Precision: {metrics['precision']:.4f}")

        # Save artifacts
        self._save_artifacts(model, metrics, calibrator, oversampling_strategy, run_optuna, calibrate)

        # MLflow logging
        self._log_mlflow(model, metrics)

        return metrics

    def _apply_oversampling(
        self, X: np.ndarray, y: np.ndarray, strategy: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply oversampling to training data only."""
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
                f"Oversampling ({strategy}): {len(X)} → {len(X_res)} samples "
                f"(positive: {y.sum()} → {y_res.sum()})"
            )
            return X_res, y_res
        except ImportError:
            logger.warning("imbalanced-learn not installed, skipping oversampling")
            return X, y

    def _calibrate_model(
        self, model: lgb.LGBMClassifier, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> CalibratedClassifierCV:
        """Apply isotonic regression calibration."""
        logger.info("Calibrating merchant model probabilities (isotonic regression)")
        calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrator.fit(X_cal, y_cal)
        return calibrator

    def _run_optuna(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> dict:
        """Run Optuna hyperparameter optimization."""
        from training.optimization.hyperopt import OptunaTuner

        tuner = OptunaTuner(domain="merchant", algorithm="lightgbm", n_trials=n_trials)
        best_params = tuner.optimize(X, y)
        tuner.save_best_params()
        return best_params

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> list[float]:
        """Stratified K-fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            model = lgb.LGBMClassifier(**self.hyperparams)
            model.fit(
                X[train_idx],
                y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            proba = model.predict_proba(X[val_idx])[:, 1]
            auc = roc_auc_score(y[val_idx], proba)
            scores.append(auc)
            logger.debug(f"Fold {fold + 1}: ROC-AUC = {auc:.4f}")

        return scores

    def _evaluate(
        self,
        model: lgb.LGBMClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        calibrator: CalibratedClassifierCV | None = None,
    ) -> dict:
        """Evaluate model on test set."""
        if calibrator is not None:
            y_proba = calibrator.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "calibrated": calibrator is not None,
        }

    def _save_artifacts(
        self,
        model: lgb.LGBMClassifier,
        metrics: dict,
        calibrator: CalibratedClassifierCV | None,
        oversampling_strategy: str | None,
        optuna_tuned: bool,
        calibrated: bool,
    ) -> None:
        """Save model, calibrator, and metadata artifacts."""
        os.makedirs(self.output_dir, exist_ok=True)

        model_path = os.path.join(self.output_dir, "model.txt")
        model.booster_.save_model(model_path)
        logger.info(f"Merchant model saved: {model_path}")

        # Save calibrator
        if calibrator is not None:
            calibrator_path = os.path.join(self.output_dir, "calibrator.pkl")
            with open(calibrator_path, "wb") as f:
                pickle.dump(calibrator, f)
            logger.info(f"Calibrator saved: {calibrator_path}")

        metadata = {
            "model_name": "pesaflow-merchant-risk",
            "version": self.model_version,
            "algorithm": "lightgbm",
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "feature_schema_hash": MERCHANT_FEATURE_SCHEMA.schema_hash,
            "hyperparameters": self.hyperparams,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "trained_at": datetime.utcnow().isoformat(),
            "oversampling_strategy": oversampling_strategy,
            "calibrated": calibrated,
            "optuna_tuned": optuna_tuned,
        }

        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _log_mlflow(self, model: lgb.LGBMClassifier, metrics: dict) -> None:
        """Log to MLflow if available."""
        try:
            import mlflow

            mlflow.set_experiment("pesaflow-merchant-risk")
            with mlflow.start_run(run_name=f"merchant-{self.model_version}"):
                safe_params = {k: str(v) for k, v in self.hyperparams.items()}
                mlflow.log_params(safe_params)
                safe_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(safe_metrics)
                mlflow.lightgbm.log_model(model, "merchant_model")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


if __name__ == "__main__":
    from training.data_generator import generate_merchant_dataset

    df = generate_merchant_dataset()
    trainer = MerchantModelTrainer()
    results = trainer.train(df)
    logger.info(f"Training complete: {results}")
