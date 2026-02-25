"""Self-Training (Pseudo-Labeling) — Iterative confidence-based SSL.

Activates at: WARMING (100+ labels)

Algorithm:
  1. Train model on labeled data only
  2. Predict on unlabeled pool
  3. Pseudo-label HIGH-CONFIDENCE samples (score > pos_threshold or < neg_threshold)
  4. Add pseudo-labeled to training set at reduced sample weight
  5. Retrain → repeat until convergence or max iterations
  6. Threshold tightening: +0.02 per iteration to prevent confirmation bias
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer, SSLConfig
from training.semi_supervised.pseudo_label_store import PseudoLabelStore


class SelfTrainingTrainer(BaseSSLTrainer):
    """Self-training pseudo-labeling with confidence thresholds."""

    technique_name = "self_training"

    def __init__(self, config: SSLConfig, db_pool=None, model_factory=None):
        super().__init__(config, db_pool)
        self._model_factory = model_factory or self._default_model_factory
        self._pseudo_store = PseudoLabelStore(db_pool) if db_pool else None

    @staticmethod
    def _default_model_factory():
        """Create a default LightGBM classifier."""
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            num_leaves=63,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=10,
            random_state=42,
            verbose=-1,
        )

    async def train(self) -> dict:
        """Run self-training loop."""
        X_labeled, y_labeled, X_unlabeled, ids_unlabeled = await self.load_data()

        if len(X_unlabeled) == 0:
            logger.info(f"[{self.technique_name}] No unlabeled data, skipping")
            return {"model": None, "iterations": 0, "converged": False}

        # Split holdout
        X_train, y_train, X_holdout, y_holdout = self.split_holdout(X_labeled, y_labeled)

        prev_auc = 0.0
        prev_pseudo_labels = np.array([])
        best_model = None
        best_auc = 0.0

        pos_threshold = self.config.positive_confidence_threshold
        neg_threshold = self.config.negative_confidence_threshold

        for iteration in range(self.config.max_iterations):
            # Train model
            model = self._model_factory()

            if iteration == 0:
                # First iteration: labeled only
                model.fit(X_train, y_train)
            else:
                # Subsequent: labeled + pseudo-labeled with sample weights
                X_combined = np.vstack([X_train, X_pseudo])
                y_combined = np.concatenate([y_train, pseudo_labels])
                weights = np.concatenate(
                    [
                        np.ones(len(X_train)),
                        np.full(len(X_pseudo), self.config.sample_weight_pseudo),
                    ]
                )
                model.fit(X_combined, y_combined, sample_weight=weights)

            # Evaluate on holdout
            holdout_probs = model.predict_proba(X_holdout)[:, 1]
            try:
                current_auc = roc_auc_score(y_holdout, holdout_probs)
            except ValueError:
                current_auc = 0.5

            logger.info(
                f"[{self.technique_name}] Iteration {iteration}: AUC={current_auc:.4f} "
                f"(pos_threshold={pos_threshold:.3f}, neg_threshold={neg_threshold:.3f})"
            )

            # Track best model
            if current_auc > best_auc:
                best_auc = current_auc
                best_model = model

            # Check convergence
            converged = iteration > 0 and self.check_convergence(prev_auc, current_auc)

            # Predict on unlabeled
            unlabeled_probs = model.predict_proba(X_unlabeled)[:, 1]

            # Select high-confidence predictions
            pos_mask = unlabeled_probs >= pos_threshold
            neg_mask = unlabeled_probs <= neg_threshold
            confident_mask = pos_mask | neg_mask

            pseudo_labels = np.where(pos_mask, 1, 0)[confident_mask]
            X_pseudo = X_unlabeled[confident_mask]
            pseudo_confidences = np.where(
                pos_mask[confident_mask],
                unlabeled_probs[confident_mask],
                1.0 - unlabeled_probs[confident_mask],
            )
            pseudo_ids = ids_unlabeled[confident_mask]

            # Compute flip rate
            flip_rate = self.compute_flip_rate(prev_pseudo_labels, pseudo_labels)

            # Safety guardrails
            safe = self.check_safety_guardrails(len(X_train), len(X_pseudo), flip_rate=flip_rate)

            # Log training run
            await self.log_training_run(
                iteration=iteration,
                labeled_count=len(X_train),
                pseudo_labeled_count=len(X_pseudo),
                flip_rate=flip_rate,
                model_auc_labeled=current_auc,
                converged=converged,
            )

            if converged or not safe or len(X_pseudo) == 0:
                break

            # Store pseudo-labels
            if self._pseudo_store and len(pseudo_ids) > 0:
                await self._pseudo_store.store_pseudo_labels(
                    domain=self.config.domain,
                    prediction_ids=pseudo_ids.tolist(),
                    features=X_pseudo,
                    pseudo_labels=pseudo_labels,
                    confidences=pseudo_confidences,
                    label_source="self_training",
                    iteration=iteration,
                )

            prev_auc = current_auc
            prev_pseudo_labels = pseudo_labels.copy()

            # Tighten thresholds
            pos_threshold = min(0.99, pos_threshold + self.config.threshold_tightening_per_iter)
            neg_threshold = max(0.01, neg_threshold - self.config.threshold_tightening_per_iter)

        return {
            "model": best_model,
            "auc_labeled": best_auc,
            "iterations": iteration + 1,
            "pseudo_labeled_count": len(X_pseudo) if "X_pseudo" in dir() else 0,
            "converged": converged if "converged" in dir() else False,
        }
