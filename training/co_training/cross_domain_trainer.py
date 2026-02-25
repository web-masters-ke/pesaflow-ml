"""Cross-Domain Co-Training — Transfer labels between fraud/AML/merchant domains.

Activates at: WARM (1K+ labels)

Uses LabelTransferEngine to propagate labels across domains with confidence decay,
then trains improved models using the augmented label sets.
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

from training.co_training.label_transfer import LabelTransferEngine
from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer, SSLConfig
from training.semi_supervised.pseudo_label_store import PseudoLabelStore

_DOMAIN_TABLE = {
    "fraud": "ml_predictions",
    "aml": "aml_predictions",
    "merchant": "merchant_risk_predictions",
}


class CrossDomainTrainer(BaseSSLTrainer):
    """Cross-domain label transfer and retraining."""

    technique_name = "cross_domain_co_training"

    def __init__(
        self,
        config: SSLConfig,
        db_pool=None,
        max_transfer_ratio: float = 0.20,
    ):
        super().__init__(config, db_pool)
        self._max_transfer_ratio = max_transfer_ratio
        self._transfer_engine = LabelTransferEngine(db_pool, max_transfer_ratio=max_transfer_ratio)
        self._pseudo_store = PseudoLabelStore(db_pool) if db_pool else None

    async def train(self) -> dict:
        """Execute cross-domain transfers then retrain the target domain model."""
        # Step 1: Execute label transfers
        transfer_results = await self._transfer_engine.execute_transfers()
        total_transferred = sum(transfer_results.values())

        logger.info(f"[{self.technique_name}] Cross-domain transfers complete: {transfer_results}")

        if total_transferred == 0:
            logger.info(f"[{self.technique_name}] No transfers executed, skipping retrain")
            return {
                "model": None,
                "iterations": 0,
                "converged": False,
                "transfers": transfer_results,
            }

        # Step 2: Load data for current domain (now includes transferred labels)
        X_labeled, y_labeled, X_unlabeled, ids_unlabeled = await self.load_data()

        if len(X_labeled) == 0:
            return {"model": None, "iterations": 0, "converged": False, "transfers": transfer_results}

        X_train, y_train, X_holdout, y_holdout = self.split_holdout(X_labeled, y_labeled)

        # Step 3: Load transferred pseudo-labels from ssl_cross_domain_transfers
        X_transferred, y_transferred, weights_transferred = await self._load_transferred_data()

        # Step 4: Train model with original + transferred labels
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            num_leaves=63,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            random_state=42,
            verbose=-1,
        )

        if len(X_transferred) > 0:
            X_combined = np.vstack([X_train, X_transferred])
            y_combined = np.concatenate([y_train, y_transferred])
            weights = np.concatenate(
                [
                    np.ones(len(X_train)),
                    weights_transferred,
                ]
            )
            model.fit(X_combined, y_combined, sample_weight=weights)
        else:
            model.fit(X_train, y_train)

        # Evaluate
        holdout_probs = model.predict_proba(X_holdout)[:, 1]
        try:
            auc = roc_auc_score(y_holdout, holdout_probs)
        except ValueError:
            auc = 0.5

        logger.info(f"[{self.technique_name}] Domain={self.config.domain}, AUC={auc:.4f}")

        await self.log_training_run(
            iteration=0,
            labeled_count=len(X_train),
            pseudo_labeled_count=len(X_transferred),
            model_auc_labeled=auc,
            converged=True,
        )

        return {
            "model": model,
            "auc_labeled": auc,
            "iterations": 1,
            "pseudo_labeled_count": len(X_transferred),
            "converged": True,
            "transfers": transfer_results,
        }

    async def _load_transferred_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load transferred labels for the current domain."""
        if not self._db:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        table = _DOMAIN_TABLE[self.config.domain]

        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT t.transferred_label, t.confidence_decay, p.feature_snapshot
                FROM ssl_cross_domain_transfers t
                JOIN {} p ON p.id::text = t.target_prediction_id
                WHERE t.target_domain = $1
                  AND p.feature_snapshot IS NOT NULL
                ORDER BY t.created_at DESC
                LIMIT 10000
                """.format(
                    table
                ),
                self.config.domain,
            )

        if not rows:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        features = []
        labels = []
        weights = []
        for row in rows:
            feat = self._extract_features(row["feature_snapshot"])
            if feat is not None:
                features.append(feat)
                labels.append(row["transferred_label"])
                weights.append(float(row["confidence_decay"]))

        return np.array(features), np.array(labels), np.array(weights)
