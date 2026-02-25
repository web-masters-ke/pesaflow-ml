"""Optuna Hyperparameter Optimization — Automated tuning for LightGBM and XGBoost models."""

import json
import os
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import xgboost as xgb
import yaml
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_engineering.feature_registry import (
    AML_FEATURE_SCHEMA,
    FRAUD_FEATURE_SCHEMA,
    MERCHANT_FEATURE_SCHEMA,
    FeatureSchema,
)

# Suppress Optuna info logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

DOMAIN_SCHEMAS = {
    "fraud": FRAUD_FEATURE_SCHEMA,
    "aml": AML_FEATURE_SCHEMA,
    "merchant": MERCHANT_FEATURE_SCHEMA,
}

# Search spaces per algorithm
LIGHTGBM_SEARCH_SPACE = {
    "num_leaves": (16, 128),
    "max_depth": (4, 12),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 1000),
    "min_child_samples": (10, 100),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 2.0),
    "reg_lambda": (0.0, 3.0),
}

XGBOOST_SEARCH_SPACE = {
    "max_depth": (4, 12),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 1000),
    "min_child_weight": (1, 20),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 2.0),
    "reg_lambda": (0.0, 3.0),
    "gamma": (0.0, 1.0),
}


class OptunaTuner:
    """Optuna-based hyperparameter tuner per domain.

    Objective: maximize ROC-AUC via 5-fold stratified CV.
    Pruning via MedianPruner for early stopping of bad trials.
    Best params auto-saved to config and logged to MLflow.
    """

    def __init__(
        self,
        domain: str,
        algorithm: str = "lightgbm",
        n_trials: int = 100,
        cv_folds: int = 5,
        mlflow_tracking_uri: str = "http://localhost:5000",
    ):
        if domain not in DOMAIN_SCHEMAS:
            raise ValueError(f"Unknown domain: {domain}")
        if algorithm not in ("lightgbm", "xgboost"):
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.domain = domain
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.schema = DOMAIN_SCHEMAS[domain]

        self.best_params: dict[str, Any] = {}
        self.best_score: float = 0.0
        self._study: optuna.Study | None = None

        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(f"pesaflow-{domain}-hyperopt")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_pos_weight: float | None = None,
    ) -> dict[str, Any]:
        """Run Optuna optimization study.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels
            scale_pos_weight: Override for class weight (computed from data if None)

        Returns:
            Best parameters dictionary
        """
        if scale_pos_weight is None:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            scale_pos_weight = float(neg_count / max(pos_count, 1))

        self._scale_pos_weight = scale_pos_weight

        logger.info(
            f"Starting Optuna optimization for {self.domain}/{self.algorithm}: "
            f"{self.n_trials} trials, {self.cv_folds}-fold CV"
        )

        self._study = optuna.create_study(
            direction="maximize",
            study_name=f"{self.domain}_{self.algorithm}_tuning",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        )

        # Store data for objective function
        self._X = X
        self._y = y

        self._study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        self.best_params = self._study.best_params
        self.best_score = self._study.best_value

        # Add fixed params
        if self.algorithm == "lightgbm":
            self.best_params.update(
                {
                    "objective": "binary",
                    "metric": "auc",
                    "boosting_type": "gbdt",
                    "scale_pos_weight": scale_pos_weight,
                    "random_state": 42,
                    "verbose": -1,
                }
            )
        else:
            self.best_params.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "scale_pos_weight": scale_pos_weight,
                    "random_state": 42,
                    "use_label_encoder": False,
                    "verbosity": 0,
                }
            )

        logger.info(f"Best ROC-AUC: {self.best_score:.4f}")
        logger.info(f"Best params: {json.dumps(self.best_params, indent=2, default=str)}")

        # Log to MLflow
        self._log_to_mlflow()

        return self.best_params

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: 5-fold CV ROC-AUC."""
        params = self._suggest_params(trial)

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(self._X, self._y)):
            X_train, X_val = self._X[train_idx], self._X[val_idx]
            y_train, y_val = self._y[train_idx], self._y[val_idx]

            if self.algorithm == "lightgbm":
                model = lgb.LGBMClassifier(
                    **params,
                    scale_pos_weight=self._scale_pos_weight,
                    random_state=42,
                    verbose=-1,
                )
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
            else:
                model = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=self._scale_pos_weight,
                    random_state=42,
                    use_label_encoder=False,
                    verbosity=0,
                )
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

            y_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            scores.append(auc)

            # Report intermediate value for pruning
            trial.report(auc, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters based on algorithm."""
        if self.algorithm == "lightgbm":
            space = LIGHTGBM_SEARCH_SPACE
            return {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"]),
                "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
                "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"], step=50),
                "min_child_samples": trial.suggest_int("min_child_samples", *space["min_child_samples"]),
                "subsample": trial.suggest_float("subsample", *space["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"]),
            }
        else:
            space = XGBOOST_SEARCH_SPACE
            return {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
                "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"], step=50),
                "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
                "subsample": trial.suggest_float("subsample", *space["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"]),
                "gamma": trial.suggest_float("gamma", *space["gamma"]),
            }

    def save_best_params(self, config_path: str = "./config/model_config.yaml") -> None:
        """Save best parameters to model_config.yaml."""
        if not self.best_params:
            logger.warning("No best params to save — run optimize() first")
            return

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

            # Map domain to config key
            config_key = {
                "fraud": "fraud_model",
                "aml": "aml_model",
                "merchant": "merchant_model",
            }.get(self.domain)

            if config_key and config_key in config:
                # Filter to only hyperparameter-relevant keys
                hyper_keys = set(LIGHTGBM_SEARCH_SPACE.keys()) | set(XGBOOST_SEARCH_SPACE.keys()) | {"scale_pos_weight"}
                tuned_params = {k: v for k, v in self.best_params.items() if k in hyper_keys}
                config[config_key]["hyperparameters"].update(tuned_params)

                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

                logger.info(f"Best params saved to {config_path} for {config_key}")
        except Exception as e:
            logger.warning(f"Failed to save params to config: {e}")

    def _log_to_mlflow(self) -> None:
        """Log optimization results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{self.domain}-{self.algorithm}-optuna"):
                # Log best params (convert all to strings for MLflow)
                safe_params = {k: str(v) for k, v in self.best_params.items()}
                mlflow.log_params(safe_params)
                mlflow.log_metric("best_cv_roc_auc", self.best_score)
                mlflow.log_metric("n_trials", self.n_trials)
                mlflow.set_tag("optimization", "optuna")
                mlflow.set_tag("domain", self.domain)
                mlflow.set_tag("algorithm", self.algorithm)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    def get_study_summary(self) -> dict:
        """Get summary of the optimization study."""
        if self._study is None:
            return {}

        return {
            "domain": self.domain,
            "algorithm": self.algorithm,
            "n_trials": len(self._study.trials),
            "n_pruned": len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_score": self.best_score,
            "best_params": self.best_params,
        }
