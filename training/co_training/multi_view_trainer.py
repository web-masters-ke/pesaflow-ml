"""Multi-View Co-Training — Train separate classifiers on feature views.

Activates at: WARM (1K+ labels)

Algorithm:
  1. Split features into independent views (transaction/user/device etc.)
  2. Train separate classifiers on each view
  3. Each view labels top-100 high-confidence predictions for other views
  4. Retrain with augmented labels → iterate 5 rounds
  5. Require 0.9 agreement threshold between views
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

from training.co_training.view_definitions import get_view_features, get_view_names
from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer, SSLConfig
from training.semi_supervised.pseudo_label_store import PseudoLabelStore


class MultiViewCoTrainer(BaseSSLTrainer):
    """Multi-view co-training using feature view splits."""

    technique_name = "multi_view_co_training"

    def __init__(
        self,
        config: SSLConfig,
        db_pool=None,
        max_rounds: int = 5,
        top_k_per_round: int = 100,
        agreement_threshold: float = 0.9,
    ):
        super().__init__(config, db_pool)
        self._max_rounds = max_rounds
        self._top_k = top_k_per_round
        self._agreement_threshold = agreement_threshold
        self._pseudo_store = PseudoLabelStore(db_pool) if db_pool else None

    async def train(self) -> dict:
        """Run multi-view co-training."""
        X_labeled, y_labeled, X_unlabeled, ids_unlabeled = await self.load_data()

        if len(X_unlabeled) == 0:
            logger.info(f"[{self.technique_name}] No unlabeled data, skipping")
            return {"model": None, "iterations": 0, "converged": False}

        X_train, y_train, X_holdout, y_holdout = self.split_holdout(X_labeled, y_labeled)

        view_names = get_view_names(self.config.domain)
        if len(view_names) < 2:
            logger.warning(f"[{self.technique_name}] Need >=2 views, got {len(view_names)}")
            return {"model": None, "iterations": 0, "converged": False}

        from lightgbm import LGBMClassifier

        # Per-view training data (starts as copy of labeled)
        view_X_train = {v: X_train.copy() for v in view_names}
        view_y_train = {v: y_train.copy() for v in view_names}

        best_auc = 0.0
        best_model = None

        for round_idx in range(self._max_rounds):
            view_models = {}
            view_predictions = {}

            # Train each view
            for view_name in view_names:
                X_view_train = get_view_features(view_X_train[view_name], self.config.domain, view_name)
                model = LGBMClassifier(
                    num_leaves=31,
                    max_depth=6,
                    learning_rate=0.05,
                    n_estimators=200,
                    random_state=42 + round_idx,
                    verbose=-1,
                )
                model.fit(X_view_train, view_y_train[view_name])
                view_models[view_name] = model

                # Predict on unlabeled
                X_view_unlabeled = get_view_features(X_unlabeled, self.config.domain, view_name)
                preds = model.predict_proba(X_view_unlabeled)[:, 1]
                view_predictions[view_name] = preds

            # Find high-confidence, high-agreement samples
            pred_matrix = np.column_stack([view_predictions[v] for v in view_names])
            mean_preds = pred_matrix.mean(axis=1)
            pred_std = pred_matrix.std(axis=1)

            # Agreement: low std = views agree
            agreement_mask = pred_std < (1.0 - self._agreement_threshold)

            # Confident + agreeing
            confident_pos = (mean_preds >= 0.9) & agreement_mask
            confident_neg = (mean_preds <= 0.1) & agreement_mask

            # Select top-K from each
            pos_indices = np.where(confident_pos)[0]
            neg_indices = np.where(confident_neg)[0]

            if len(pos_indices) > self._top_k:
                top_pos_idx = pos_indices[np.argsort(mean_preds[pos_indices])[::-1][: self._top_k]]
            else:
                top_pos_idx = pos_indices

            if len(neg_indices) > self._top_k:
                top_neg_idx = neg_indices[np.argsort(mean_preds[neg_indices])[: self._top_k]]
            else:
                top_neg_idx = neg_indices

            selected_idx = np.concatenate([top_pos_idx, top_neg_idx])
            if len(selected_idx) == 0:
                logger.info(f"[{self.technique_name}] No agreeable samples at round {round_idx}")
                break

            selected_labels = np.where(mean_preds[selected_idx] >= 0.5, 1, 0)

            # Add to each view's training data
            for view_name in view_names:
                view_X_train[view_name] = np.vstack([view_X_train[view_name], X_unlabeled[selected_idx]])
                view_y_train[view_name] = np.concatenate([view_y_train[view_name], selected_labels])

            # Safety check
            safe = self.check_safety_guardrails(len(X_train), len(selected_idx) * (round_idx + 1))

            # Evaluate ensemble (average of all views on full features)
            # Train a full-feature model on combined data
            X_full_train = np.vstack([X_train, X_unlabeled[selected_idx]])
            y_full_train = np.concatenate([y_train, selected_labels])
            weights = np.concatenate(
                [
                    np.ones(len(X_train)),
                    np.full(len(selected_idx), self.config.sample_weight_pseudo),
                ]
            )

            full_model = LGBMClassifier(
                num_leaves=63,
                max_depth=8,
                learning_rate=0.05,
                n_estimators=500,
                random_state=42,
                verbose=-1,
            )
            full_model.fit(X_full_train, y_full_train, sample_weight=weights)

            holdout_probs = full_model.predict_proba(X_holdout)[:, 1]
            try:
                current_auc = roc_auc_score(y_holdout, holdout_probs)
            except ValueError:
                current_auc = 0.5

            logger.info(
                f"[{self.technique_name}] Round {round_idx}: AUC={current_auc:.4f}, "
                f"added={len(selected_idx)} (pos={len(top_pos_idx)}, neg={len(top_neg_idx)})"
            )

            if current_auc > best_auc:
                best_auc = current_auc
                best_model = full_model

            if not safe:
                break

            # Remove selected from unlabeled pool
            remaining_mask = np.ones(len(X_unlabeled), dtype=bool)
            remaining_mask[selected_idx] = False
            X_unlabeled = X_unlabeled[remaining_mask]
            ids_unlabeled = ids_unlabeled[remaining_mask]

        # Store pseudo-labels
        if self._pseudo_store and "selected_idx" in dir() and len(selected_idx) > 0:
            selected_confidences = np.abs(mean_preds[selected_idx] - 0.5) + 0.5
            await self._pseudo_store.store_pseudo_labels(
                domain=self.config.domain,
                prediction_ids=(
                    ids_unlabeled[: len(selected_idx)].tolist() if len(ids_unlabeled) >= len(selected_idx) else []
                ),
                features=(
                    X_unlabeled[: len(selected_idx)]
                    if len(X_unlabeled) >= len(selected_idx)
                    else np.empty((0, X_train.shape[1]))
                ),
                pseudo_labels=selected_labels,
                confidences=selected_confidences,
                label_source="co_training",
            )

        await self.log_training_run(
            iteration=round_idx if "round_idx" in dir() else 0,
            labeled_count=len(X_train),
            pseudo_labeled_count=len(view_X_train[view_names[0]]) - len(X_train),
            model_auc_labeled=best_auc,
            converged=True,
        )

        return {
            "model": best_model,
            "auc_labeled": best_auc,
            "iterations": (round_idx + 1) if "round_idx" in dir() else 0,
            "pseudo_labeled_count": len(view_X_train[view_names[0]]) - len(X_train),
            "converged": True,
        }
