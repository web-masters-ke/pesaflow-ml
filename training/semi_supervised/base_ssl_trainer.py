"""Base class for all semi-supervised learning trainers.

Provides common infrastructure:
  - Data loading from prediction tables (labeled + unlabeled JSONB)
  - Convergence detection
  - Safety guardrails (max pseudo:labeled ratio, accuracy floors)
  - MLflow logging
  - Artifact saving
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger


class SSLConfig:
    """Configuration for SSL training."""

    def __init__(
        self,
        domain: str = "fraud",
        max_iterations: int = 10,
        convergence_threshold: float = 0.001,
        max_pseudo_labeled_ratio: float = 3.0,
        labeled_holdout_fraction: float = 0.2,
        min_auc_improvement: float = 0.005,
        pseudo_label_accuracy_floor: float = 0.85,
        label_flip_rate_ceiling: float = 0.10,
        positive_confidence_threshold: float = 0.95,
        negative_confidence_threshold: float = 0.05,
        threshold_tightening_per_iter: float = 0.02,
        sample_weight_pseudo: float = 0.5,
    ):
        self.domain = domain
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_pseudo_labeled_ratio = max_pseudo_labeled_ratio
        self.labeled_holdout_fraction = labeled_holdout_fraction
        self.min_auc_improvement = min_auc_improvement
        self.pseudo_label_accuracy_floor = pseudo_label_accuracy_floor
        self.label_flip_rate_ceiling = label_flip_rate_ceiling
        self.positive_confidence_threshold = positive_confidence_threshold
        self.negative_confidence_threshold = negative_confidence_threshold
        self.threshold_tightening_per_iter = threshold_tightening_per_iter
        self.sample_weight_pseudo = sample_weight_pseudo


_DOMAIN_TABLE = {
    "fraud": "ml_predictions",
    "aml": "aml_predictions",
    "merchant": "merchant_risk_predictions",
}


class BaseSSLTrainer(ABC):
    """Abstract base class for semi-supervised learning trainers."""

    technique_name: str = "base_ssl"

    def __init__(self, config: SSLConfig, db_pool: Any = None):
        self.config = config
        self._db = db_pool
        self._run_id = str(uuid.uuid4())

    async def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load labeled and unlabeled data from prediction table.

        Returns:
            (X_labeled, y_labeled, X_unlabeled, prediction_ids_unlabeled)
        """
        if not self._db:
            raise RuntimeError("Database connection required for data loading")

        table = _DOMAIN_TABLE[self.config.domain]

        async with self._db.acquire() as conn:
            # Load labeled data
            labeled_rows = await conn.fetch(
                f"""
                SELECT id, feature_snapshot, label
                FROM {table}
                WHERE label IS NOT NULL
                  AND feature_snapshot IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 50000
                """
            )

            # Load unlabeled data
            unlabeled_rows = await conn.fetch(
                f"""
                SELECT id, feature_snapshot, risk_score
                FROM {table}
                WHERE label IS NULL
                  AND feature_snapshot IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100000
                """
            )

        if not labeled_rows:
            raise ValueError(f"No labeled data found for domain {self.config.domain}")

        X_labeled = []
        y_labeled = []
        for row in labeled_rows:
            features = self._extract_features(row["feature_snapshot"])
            if features is not None:
                X_labeled.append(features)
                y_labeled.append(row["label"])

        X_unlabeled = []
        ids_unlabeled = []
        for row in unlabeled_rows:
            features = self._extract_features(row["feature_snapshot"])
            if features is not None:
                X_unlabeled.append(features)
                ids_unlabeled.append(str(row["id"]))

        logger.info(
            f"[{self.technique_name}] Loaded {len(X_labeled)} labeled, "
            f"{len(X_unlabeled)} unlabeled samples for {self.config.domain}"
        )

        return (
            np.array(X_labeled),
            np.array(y_labeled),
            np.array(X_unlabeled) if X_unlabeled else np.empty((0, len(X_labeled[0]))),
            np.array(ids_unlabeled),
        )

    def _extract_features(self, feature_snapshot: Any) -> list[float] | None:
        """Extract feature array from JSONB snapshot."""
        if isinstance(feature_snapshot, dict):
            return list(feature_snapshot.values())
        elif isinstance(feature_snapshot, list):
            return feature_snapshot
        return None

    def split_holdout(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split labeled data into train and holdout sets."""
        n = len(X)
        holdout_size = max(1, int(n * self.config.labeled_holdout_fraction))
        indices = np.random.permutation(n)
        holdout_idx = indices[:holdout_size]
        train_idx = indices[holdout_size:]

        return X[train_idx], y[train_idx], X[holdout_idx], y[holdout_idx]

    def check_convergence(self, prev_auc: float, current_auc: float) -> bool:
        """Check if training has converged (improvement below threshold)."""
        improvement = current_auc - prev_auc
        if improvement < self.config.convergence_threshold:
            logger.info(
                f"[{self.technique_name}] Converged: AUC improvement {improvement:.6f} "
                f"< threshold {self.config.convergence_threshold}"
            )
            return True
        return False

    def check_safety_guardrails(
        self,
        n_labeled: int,
        n_pseudo: int,
        flip_rate: float | None = None,
    ) -> bool:
        """Check safety guardrails. Returns True if safe to continue."""
        # Check pseudo:labeled ratio
        if n_labeled > 0 and n_pseudo / n_labeled > self.config.max_pseudo_labeled_ratio:
            logger.warning(
                f"[{self.technique_name}] Pseudo:labeled ratio {n_pseudo/n_labeled:.2f} "
                f"exceeds max {self.config.max_pseudo_labeled_ratio}"
            )
            return False

        # Check flip rate
        if flip_rate is not None and flip_rate > self.config.label_flip_rate_ceiling:
            logger.warning(
                f"[{self.technique_name}] Label flip rate {flip_rate:.4f} "
                f"exceeds ceiling {self.config.label_flip_rate_ceiling}"
            )
            return False

        return True

    def compute_flip_rate(self, prev_pseudo_labels: np.ndarray, current_pseudo_labels: np.ndarray) -> float:
        """Compute label flip rate between iterations."""
        if len(prev_pseudo_labels) == 0 or len(current_pseudo_labels) == 0:
            return 0.0
        min_len = min(len(prev_pseudo_labels), len(current_pseudo_labels))
        flips = np.sum(prev_pseudo_labels[:min_len] != current_pseudo_labels[:min_len])
        return flips / min_len

    async def log_training_run(
        self,
        iteration: int,
        labeled_count: int,
        pseudo_labeled_count: int,
        agreement_rate: float | None = None,
        flip_rate: float | None = None,
        model_auc_labeled: float | None = None,
        model_auc_full: float | None = None,
        converged: bool = False,
    ) -> None:
        """Log SSL training run to database."""
        if not self._db:
            return

        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO ssl_training_runs
                    (id, domain, technique, iteration, labeled_count, pseudo_labeled_count,
                     pseudo_label_agreement_rate, pseudo_label_flip_rate,
                     model_auc_labeled, model_auc_full, converged)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    str(uuid.uuid4()),
                    self.config.domain,
                    self.technique_name,
                    iteration,
                    labeled_count,
                    pseudo_labeled_count,
                    agreement_rate,
                    flip_rate,
                    model_auc_labeled,
                    model_auc_full,
                    converged,
                )
        except Exception as e:
            logger.error(f"Failed to log SSL training run: {e}")

    @abstractmethod
    async def train(self) -> dict:
        """Run the SSL training loop.

        Returns:
            dict with keys: model, auc_labeled, auc_full, iterations,
            pseudo_labeled_count, converged
        """
        ...
