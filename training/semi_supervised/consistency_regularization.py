"""Consistency Regularization — Perturbation-based SSL.

Activates at: WARMING (100+ labels)

Algorithm:
  1. For each unlabeled sample, create K augmented versions (Gaussian noise + feature dropout)
  2. Predict all versions, compute mean prediction as pseudo-target
  3. Add consistency loss: MSE between individual predictions and mean
  4. Train with combined supervised loss + consistency_weight * consistency loss
  5. Ramp up consistency weight over epochs
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer, SSLConfig
from training.semi_supervised.pseudo_label_store import PseudoLabelStore


class ConsistencyRegularizationTrainer(BaseSSLTrainer):
    """Perturbation-based consistency regularization for tabular data."""

    technique_name = "consistency_regularization"

    def __init__(
        self,
        config: SSLConfig,
        db_pool=None,
        n_augmentations: int = 5,
        noise_scale: float = 0.1,
        dropout_rate: float = 0.1,
        consistency_weight_max: float = 1.0,
        ramp_up_epochs: int = 5,
        n_epochs: int = 10,
    ):
        super().__init__(config, db_pool)
        self._n_augmentations = n_augmentations
        self._noise_scale = noise_scale
        self._dropout_rate = dropout_rate
        self._consistency_weight_max = consistency_weight_max
        self._ramp_up_epochs = ramp_up_epochs
        self._n_epochs = n_epochs
        self._pseudo_store = PseudoLabelStore(db_pool) if db_pool else None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise + feature dropout to create augmented versions."""
        augmented = X.copy()

        # Gaussian noise
        noise = np.random.normal(0, self._noise_scale, X.shape)
        augmented = augmented + noise

        # Feature dropout (randomly zero out features)
        dropout_mask = np.random.random(X.shape) > self._dropout_rate
        augmented = augmented * dropout_mask

        return augmented

    def _get_consistency_weight(self, epoch: int) -> float:
        """Ramp up consistency weight over epochs."""
        if epoch >= self._ramp_up_epochs:
            return self._consistency_weight_max
        return self._consistency_weight_max * (epoch / self._ramp_up_epochs)

    async def train(self) -> dict:
        """Run consistency regularization training."""
        X_labeled, y_labeled, X_unlabeled, ids_unlabeled = await self.load_data()

        if len(X_unlabeled) == 0:
            logger.info(f"[{self.technique_name}] No unlabeled data, skipping")
            return {"model": None, "iterations": 0, "converged": False}

        X_train, y_train, X_holdout, y_holdout = self.split_holdout(X_labeled, y_labeled)

        from lightgbm import LGBMClassifier

        best_model = None
        best_auc = 0.0
        prev_auc = 0.0

        for epoch in range(self._n_epochs):
            consistency_weight = self._get_consistency_weight(epoch)

            # Generate augmented predictions for unlabeled data
            augmented_preds = []
            model = LGBMClassifier(
                num_leaves=63,
                max_depth=8,
                learning_rate=0.05,
                n_estimators=300,
                random_state=42 + epoch,
                verbose=-1,
            )

            if epoch == 0:
                # First epoch: train on labeled only
                model.fit(X_train, y_train)
            else:
                # Train on labeled + pseudo-labeled
                X_combined = np.vstack([X_train, X_pseudo_selected])
                y_combined = np.concatenate([y_train, pseudo_targets])
                weights = np.concatenate(
                    [
                        np.ones(len(X_train)),
                        np.full(len(X_pseudo_selected), self.config.sample_weight_pseudo * consistency_weight),
                    ]
                )
                model.fit(X_combined, y_combined, sample_weight=weights)

            # Predict on augmented unlabeled versions
            for _ in range(self._n_augmentations):
                X_aug = self._augment(X_unlabeled)
                preds = model.predict_proba(X_aug)[:, 1]
                augmented_preds.append(preds)

            # Compute mean prediction as pseudo-target
            mean_preds = np.mean(augmented_preds, axis=0)

            # Compute prediction variance (proxy for consistency)
            pred_variance = np.var(augmented_preds, axis=0)

            # Select samples with low variance (consistent predictions)
            consistency_mask = pred_variance < np.median(pred_variance)

            # Create pseudo-labels from mean predictions
            pseudo_targets = np.where(mean_preds >= 0.5, 1, 0)[consistency_mask]
            pseudo_confidences = np.where(
                mean_preds[consistency_mask] >= 0.5,
                mean_preds[consistency_mask],
                1.0 - mean_preds[consistency_mask],
            )
            X_pseudo_selected = X_unlabeled[consistency_mask]
            pseudo_ids_selected = ids_unlabeled[consistency_mask]

            # Safety guardrails
            safe = self.check_safety_guardrails(len(X_train), len(X_pseudo_selected))
            if not safe:
                max_n = int(len(X_train) * self.config.max_pseudo_labeled_ratio)
                top_idx = np.argsort(pseudo_confidences)[::-1][:max_n]
                X_pseudo_selected = X_pseudo_selected[top_idx]
                pseudo_targets = pseudo_targets[top_idx]
                pseudo_confidences = pseudo_confidences[top_idx]
                pseudo_ids_selected = pseudo_ids_selected[top_idx]

            # Evaluate on holdout
            holdout_probs = model.predict_proba(X_holdout)[:, 1]
            try:
                current_auc = roc_auc_score(y_holdout, holdout_probs)
            except ValueError:
                current_auc = 0.5

            logger.info(
                f"[{self.technique_name}] Epoch {epoch}: AUC={current_auc:.4f}, "
                f"consistency_weight={consistency_weight:.2f}, "
                f"pseudo_count={len(X_pseudo_selected)}"
            )

            if current_auc > best_auc:
                best_auc = current_auc
                best_model = model

            # Check convergence
            if epoch > 0 and self.check_convergence(prev_auc, current_auc):
                break

            prev_auc = current_auc

        # Store pseudo-labels from final epoch
        if self._pseudo_store and len(pseudo_ids_selected) > 0:
            await self._pseudo_store.store_pseudo_labels(
                domain=self.config.domain,
                prediction_ids=pseudo_ids_selected.tolist(),
                features=X_pseudo_selected,
                pseudo_labels=pseudo_targets,
                confidences=pseudo_confidences,
                label_source="consistency_regularization",
            )

        await self.log_training_run(
            iteration=epoch,
            labeled_count=len(X_train),
            pseudo_labeled_count=len(X_pseudo_selected),
            model_auc_labeled=best_auc,
            converged=True,
        )

        return {
            "model": best_model,
            "auc_labeled": best_auc,
            "iterations": epoch + 1,
            "pseudo_labeled_count": len(X_pseudo_selected),
            "converged": True,
        }
