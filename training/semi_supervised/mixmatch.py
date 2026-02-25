"""MixMatch — Augmentation + Sharpening + MixUp for tabular SSL.

Activates at: WARMING (100+ labels)

Algorithm:
  1. For each unlabeled x: generate K=3 augmented versions
  2. Average predictions, sharpen with temperature T=0.5
  3. MixUp: interpolate labeled + pseudo-labeled examples using Beta(0.75, 0.75)
  4. Train on mixed batch with separate labeled/unlabeled loss terms
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer, SSLConfig
from training.semi_supervised.pseudo_label_store import PseudoLabelStore


class MixMatchTrainer(BaseSSLTrainer):
    """MixMatch-style SSL adapted for tabular data."""

    technique_name = "mixmatch"

    def __init__(
        self,
        config: SSLConfig,
        db_pool=None,
        n_augmentations: int = 3,
        temperature: float = 0.5,
        mixup_alpha: float = 0.75,
        noise_scale: float = 0.1,
        dropout_rate: float = 0.15,
    ):
        super().__init__(config, db_pool)
        self._n_augmentations = n_augmentations
        self._temperature = temperature
        self._mixup_alpha = mixup_alpha
        self._noise_scale = noise_scale
        self._dropout_rate = dropout_rate
        self._pseudo_store = PseudoLabelStore(db_pool) if db_pool else None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Augment tabular data with noise + feature dropout."""
        noise = np.random.normal(0, self._noise_scale, X.shape)
        augmented = X + noise
        dropout_mask = np.random.random(X.shape) > self._dropout_rate
        return augmented * dropout_mask

    def _sharpen(self, probs: np.ndarray) -> np.ndarray:
        """Sharpen probability distribution using temperature scaling."""
        # For binary: sharpen pushes predictions towards 0 or 1
        sharpened = probs ** (1.0 / self._temperature)
        # Normalize (for binary classification)
        return sharpened / (sharpened + (1.0 - probs) ** (1.0 / self._temperature))

    def _mixup(
        self,
        X1: np.ndarray,
        y1: np.ndarray,
        X2: np.ndarray,
        y2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """MixUp interpolation between two datasets."""
        n = min(len(X1), len(X2))
        if n == 0:
            return X1, y1

        X1, y1 = X1[:n], y1[:n]
        X2, y2 = X2[:n], y2[:n]

        # Sample mixing coefficients from Beta distribution
        lam = np.random.beta(self._mixup_alpha, self._mixup_alpha, size=n)
        lam = np.maximum(lam, 1.0 - lam)  # Ensure lam >= 0.5 (closer to X1)

        X_mixed = lam[:, np.newaxis] * X1 + (1.0 - lam[:, np.newaxis]) * X2
        y_mixed = lam * y1 + (1.0 - lam) * y2

        return X_mixed, y_mixed

    async def train(self) -> dict:
        """Run MixMatch training loop."""
        X_labeled, y_labeled, X_unlabeled, ids_unlabeled = await self.load_data()

        if len(X_unlabeled) == 0:
            logger.info(f"[{self.technique_name}] No unlabeled data, skipping")
            return {"model": None, "iterations": 0, "converged": False}

        X_train, y_train, X_holdout, y_holdout = self.split_holdout(X_labeled, y_labeled)

        from lightgbm import LGBMClassifier

        best_model = None
        best_auc = 0.0
        prev_auc = 0.0

        for iteration in range(self.config.max_iterations):
            # Step 1: Train current model
            model = LGBMClassifier(
                num_leaves=63,
                max_depth=8,
                learning_rate=0.05,
                n_estimators=300,
                random_state=42 + iteration,
                verbose=-1,
            )

            if iteration == 0:
                model.fit(X_train, y_train)
            else:
                model.fit(X_mixed_all, y_mixed_all, sample_weight=weights_all)

            # Step 2: Generate augmented predictions for unlabeled data
            augmented_preds = []
            for _ in range(self._n_augmentations):
                X_aug = self._augment(X_unlabeled)
                preds = model.predict_proba(X_aug)[:, 1]
                augmented_preds.append(preds)

            # Step 3: Average and sharpen
            avg_preds = np.mean(augmented_preds, axis=0)
            sharpened_preds = self._sharpen(avg_preds)

            # Step 4: Create pseudo-labels
            pseudo_labels = sharpened_preds  # Soft labels for MixUp
            pseudo_hard = (sharpened_preds >= 0.5).astype(int)

            # Select confident samples for mixing
            confident_mask = (sharpened_preds >= 0.8) | (sharpened_preds <= 0.2)
            X_pseudo_confident = X_unlabeled[confident_mask]
            y_pseudo_confident = sharpened_preds[confident_mask]

            if len(X_pseudo_confident) == 0:
                logger.info(f"[{self.technique_name}] No confident pseudo-labels at iteration {iteration}")
                break

            # Safety check
            safe = self.check_safety_guardrails(len(X_train), len(X_pseudo_confident))
            if not safe:
                max_n = int(len(X_train) * self.config.max_pseudo_labeled_ratio)
                confidences = np.abs(y_pseudo_confident - 0.5)
                top_idx = np.argsort(confidences)[::-1][:max_n]
                X_pseudo_confident = X_pseudo_confident[top_idx]
                y_pseudo_confident = y_pseudo_confident[top_idx]

            # Step 5: MixUp labeled + pseudo-labeled
            # Shuffle pseudo for mixing
            shuffle_idx = np.random.permutation(len(X_pseudo_confident))
            X_pseudo_shuffled = X_pseudo_confident[shuffle_idx]
            y_pseudo_shuffled = y_pseudo_confident[shuffle_idx]

            X_mixed_labeled, y_mixed_labeled = self._mixup(
                X_train,
                y_train.astype(float),
                X_pseudo_shuffled[: len(X_train)],
                y_pseudo_shuffled[: len(X_train)],
            )

            X_mixed_unlabeled, y_mixed_unlabeled = self._mixup(
                X_pseudo_confident,
                y_pseudo_confident,
                X_train[np.random.choice(len(X_train), len(X_pseudo_confident))],
                y_train[np.random.choice(len(X_train), len(X_pseudo_confident))].astype(float),
            )

            # Combine
            X_mixed_all = np.vstack([X_mixed_labeled, X_mixed_unlabeled])
            y_mixed_soft = np.concatenate([y_mixed_labeled, y_mixed_unlabeled])
            # Round soft labels for tree-based classifier
            y_mixed_all = (y_mixed_soft >= 0.5).astype(int)

            weights_all = np.concatenate(
                [
                    np.ones(len(X_mixed_labeled)),
                    np.full(len(X_mixed_unlabeled), self.config.sample_weight_pseudo),
                ]
            )

            # Evaluate on holdout
            holdout_probs = model.predict_proba(X_holdout)[:, 1]
            try:
                current_auc = roc_auc_score(y_holdout, holdout_probs)
            except ValueError:
                current_auc = 0.5

            logger.info(
                f"[{self.technique_name}] Iteration {iteration}: AUC={current_auc:.4f}, "
                f"pseudo_count={len(X_pseudo_confident)}"
            )

            if current_auc > best_auc:
                best_auc = current_auc
                best_model = model

            if iteration > 0 and self.check_convergence(prev_auc, current_auc):
                break

            prev_auc = current_auc

        # Store pseudo-labels from final iteration
        if self._pseudo_store and "confident_mask" in dir():
            final_ids = ids_unlabeled[confident_mask]
            final_confidences = np.abs(sharpened_preds[confident_mask] - 0.5) + 0.5

            if len(final_ids) > 0:
                await self._pseudo_store.store_pseudo_labels(
                    domain=self.config.domain,
                    prediction_ids=final_ids.tolist(),
                    features=X_unlabeled[confident_mask],
                    pseudo_labels=pseudo_hard[confident_mask],
                    confidences=final_confidences,
                    label_source="mixmatch",
                    iteration=iteration,
                )

        await self.log_training_run(
            iteration=iteration if "iteration" in dir() else 0,
            labeled_count=len(X_train),
            pseudo_labeled_count=len(X_pseudo_confident) if "X_pseudo_confident" in dir() else 0,
            model_auc_labeled=best_auc,
            converged=True,
        )

        return {
            "model": best_model,
            "auc_labeled": best_auc,
            "iterations": (iteration + 1) if "iteration" in dir() else 0,
            "pseudo_labeled_count": len(X_pseudo_confident) if "X_pseudo_confident" in dir() else 0,
            "converged": True,
        }
