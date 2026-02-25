"""Label Propagation — Graph-based label spreading.

Activates at: WARMING (100+ labels)

Algorithm:
  1. Build k-NN graph (k=10) from feature vectors
  2. Use sklearn LabelSpreading(kernel='knn', alpha=0.2)
  3. Propagate known labels through graph to nearby unlabeled points
  4. Accept propagated labels where confidence > 0.7
  5. Especially powerful for AML (network/device cluster features encode proximity)
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer, SSLConfig
from training.semi_supervised.pseudo_label_store import PseudoLabelStore


class LabelPropagationTrainer(BaseSSLTrainer):
    """Graph-based label spreading using k-NN similarity."""

    technique_name = "label_propagation"

    def __init__(
        self,
        config: SSLConfig,
        db_pool=None,
        n_neighbors: int = 10,
        alpha: float = 0.2,
        propagation_confidence_threshold: float = 0.7,
    ):
        super().__init__(config, db_pool)
        self._n_neighbors = n_neighbors
        self._alpha = alpha
        self._confidence_threshold = propagation_confidence_threshold
        self._pseudo_store = PseudoLabelStore(db_pool) if db_pool else None

    async def train(self) -> dict:
        """Run label propagation."""
        X_labeled, y_labeled, X_unlabeled, ids_unlabeled = await self.load_data()

        if len(X_unlabeled) == 0:
            logger.info(f"[{self.technique_name}] No unlabeled data, skipping")
            return {"model": None, "iterations": 0, "converged": False}

        # Split holdout
        X_train, y_train, X_holdout, y_holdout = self.split_holdout(X_labeled, y_labeled)

        # Combine all data (labeled + unlabeled)
        X_all = np.vstack([X_train, X_unlabeled])
        y_all = np.concatenate(
            [
                y_train,
                np.full(len(X_unlabeled), -1),  # -1 = unlabeled
            ]
        )

        # Scale features for better k-NN distance computation
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        X_holdout_scaled = scaler.transform(X_holdout)

        # Cap n_neighbors to avoid issues with small datasets
        n_neighbors = min(self._n_neighbors, len(X_all) - 1)

        logger.info(
            f"[{self.technique_name}] Running label spreading on {len(X_train)} labeled "
            f"+ {len(X_unlabeled)} unlabeled, k={n_neighbors}, alpha={self._alpha}"
        )

        # Run label spreading
        label_spreader = LabelSpreading(
            kernel="knn",
            n_neighbors=n_neighbors,
            alpha=self._alpha,
            max_iter=100,
        )
        label_spreader.fit(X_all_scaled, y_all)

        # Get propagated label probabilities
        propagated_probs = label_spreader.label_distributions_

        # Extract only the unlabeled portion
        unlabeled_start = len(X_train)
        unlabeled_probs = propagated_probs[unlabeled_start:]

        # Determine propagated labels and confidences
        propagated_labels = np.argmax(unlabeled_probs, axis=1)
        propagated_confidences = np.max(unlabeled_probs, axis=1)

        # Filter by confidence threshold
        confident_mask = propagated_confidences >= self._confidence_threshold
        X_pseudo = X_unlabeled[confident_mask]
        pseudo_labels = propagated_labels[confident_mask]
        pseudo_confidences = propagated_confidences[confident_mask]
        pseudo_ids = ids_unlabeled[confident_mask]

        logger.info(
            f"[{self.technique_name}] Propagated {len(pseudo_labels)} labels "
            f"(confidence >= {self._confidence_threshold})"
        )

        # Safety check
        safe = self.check_safety_guardrails(len(X_train), len(X_pseudo))
        if not safe:
            # Reduce to max ratio
            max_pseudo = int(len(X_train) * self.config.max_pseudo_labeled_ratio)
            top_idx = np.argsort(pseudo_confidences)[::-1][:max_pseudo]
            X_pseudo = X_pseudo[top_idx]
            pseudo_labels = pseudo_labels[top_idx]
            pseudo_confidences = pseudo_confidences[top_idx]
            pseudo_ids = pseudo_ids[top_idx]

        # Train a proper classifier on labeled + propagated
        from lightgbm import LGBMClassifier

        final_model = LGBMClassifier(
            num_leaves=63,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            random_state=42,
            verbose=-1,
        )

        if len(X_pseudo) > 0:
            X_combined = np.vstack([X_train, X_pseudo])
            y_combined = np.concatenate([y_train, pseudo_labels])
            weights = np.concatenate(
                [
                    np.ones(len(X_train)),
                    np.full(len(X_pseudo), self.config.sample_weight_pseudo),
                ]
            )
            final_model.fit(X_combined, y_combined, sample_weight=weights)
        else:
            final_model.fit(X_train, y_train)

        # Evaluate
        holdout_probs = final_model.predict_proba(X_holdout)[:, 1]
        try:
            auc = roc_auc_score(y_holdout, holdout_probs)
        except ValueError:
            auc = 0.5

        # Store pseudo-labels
        if self._pseudo_store and len(pseudo_ids) > 0:
            await self._pseudo_store.store_pseudo_labels(
                domain=self.config.domain,
                prediction_ids=pseudo_ids.tolist(),
                features=X_pseudo,
                pseudo_labels=pseudo_labels,
                confidences=pseudo_confidences,
                label_source="label_propagation",
            )

        # Log run
        await self.log_training_run(
            iteration=0,
            labeled_count=len(X_train),
            pseudo_labeled_count=len(X_pseudo),
            model_auc_labeled=auc,
            converged=True,
        )

        logger.info(f"[{self.technique_name}] Final AUC={auc:.4f}")

        return {
            "model": final_model,
            "auc_labeled": auc,
            "iterations": 1,
            "pseudo_labeled_count": len(X_pseudo),
            "converged": True,
        }
