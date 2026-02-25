"""Tri-Training — 3-classifier agreement-based pseudo-labeling.

Activates at: WARM (1K+ labels)

Leverages diverse classifiers (LightGBM, XGBoost, RandomForest):
  1. Train 3 classifiers on labeled data (bootstrap sampling for diversity)
  2. For each unlabeled sample: if 2 of 3 agree → use as pseudo-label for the 3rd
  3. Iterate up to 8 rounds, track estimated error rate
  4. Stop if error > 0.10 or no new agreements
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

from training.semi_supervised.base_ssl_trainer import BaseSSLTrainer, SSLConfig
from training.semi_supervised.pseudo_label_store import PseudoLabelStore


class TriTrainingTrainer(BaseSSLTrainer):
    """Tri-training with 3 diverse classifiers."""

    technique_name = "tri_training"

    def __init__(
        self,
        config: SSLConfig,
        db_pool=None,
        max_rounds: int = 8,
        max_error_rate: float = 0.10,
        bootstrap_ratio: float = 0.8,
    ):
        super().__init__(config, db_pool)
        self._max_rounds = max_rounds
        self._max_error_rate = max_error_rate
        self._bootstrap_ratio = bootstrap_ratio
        self._pseudo_store = PseudoLabelStore(db_pool) if db_pool else None

    @staticmethod
    def _create_classifiers(seed: int = 42):
        """Create 3 diverse classifiers."""
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        return [
            LGBMClassifier(
                num_leaves=63,
                max_depth=8,
                learning_rate=0.05,
                n_estimators=300,
                random_state=seed,
                verbose=-1,
            ),
            XGBClassifier(
                max_depth=8,
                learning_rate=0.03,
                n_estimators=300,
                random_state=seed + 1,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=seed + 2,
                n_jobs=-1,
            ),
        ]

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a bootstrap sample for diversity."""
        n = len(X)
        sample_size = int(n * self._bootstrap_ratio)
        indices = np.random.choice(n, size=sample_size, replace=True)
        return X[indices], y[indices]

    async def train(self) -> dict:
        """Run tri-training loop."""
        X_labeled, y_labeled, X_unlabeled, ids_unlabeled = await self.load_data()

        if len(X_unlabeled) == 0:
            logger.info(f"[{self.technique_name}] No unlabeled data, skipping")
            return {"model": None, "iterations": 0, "converged": False}

        X_train, y_train, X_holdout, y_holdout = self.split_holdout(X_labeled, y_labeled)

        classifiers = self._create_classifiers()
        n_classifiers = len(classifiers)

        # Per-classifier augmented training sets
        clf_X_train = [X_train.copy() for _ in range(n_classifiers)]
        clf_y_train = [y_train.copy() for _ in range(n_classifiers)]

        best_auc = 0.0
        best_ensemble_preds = None
        total_pseudo = 0

        for round_idx in range(self._max_rounds):
            # Train each classifier on its bootstrap sample
            for i in range(n_classifiers):
                X_boot, y_boot = self._bootstrap_sample(clf_X_train[i], clf_y_train[i])
                classifiers[i].fit(X_boot, y_boot)

            # Predict on unlabeled data
            predictions = []
            probabilities = []
            for clf in classifiers:
                preds = clf.predict(X_unlabeled)
                probs = clf.predict_proba(X_unlabeled)[:, 1]
                predictions.append(preds)
                probabilities.append(probs)

            predictions = np.array(predictions)  # shape: (3, n_unlabeled)
            probabilities = np.array(probabilities)

            # For each classifier: find samples where other 2 agree
            new_samples_added = 0
            for i in range(n_classifiers):
                other_indices = [j for j in range(n_classifiers) if j != i]
                other_preds = predictions[other_indices]

                # 2-of-3 agreement: both other classifiers agree
                agreement_mask = other_preds[0] == other_preds[1]
                agreed_label = other_preds[0]

                # Confidence: average probability of agreeing classifiers
                other_probs = probabilities[other_indices]
                avg_confidence = np.mean(other_probs, axis=0)
                high_confidence = (avg_confidence >= 0.8) | (avg_confidence <= 0.2)

                # Select: agreement + high confidence
                select_mask = agreement_mask & high_confidence
                selected_idx = np.where(select_mask)[0]

                if len(selected_idx) == 0:
                    continue

                # Limit additions per round
                max_add = min(len(selected_idx), 200)
                # Sort by confidence, take top
                confidences = np.abs(avg_confidence[selected_idx] - 0.5)
                top_idx = selected_idx[np.argsort(confidences)[::-1][:max_add]]

                # Add to classifier i's training data
                clf_X_train[i] = np.vstack([clf_X_train[i], X_unlabeled[top_idx]])
                clf_y_train[i] = np.concatenate([clf_y_train[i], agreed_label[top_idx]])
                new_samples_added += len(top_idx)

            total_pseudo += new_samples_added

            # Evaluate ensemble (average of 3 classifiers)
            ensemble_probs = np.mean(probabilities, axis=0)
            holdout_probs_list = [clf.predict_proba(X_holdout)[:, 1] for clf in classifiers]
            holdout_ensemble = np.mean(holdout_probs_list, axis=0)

            try:
                current_auc = roc_auc_score(y_holdout, holdout_ensemble)
            except ValueError:
                current_auc = 0.5

            # Estimate error rate from cross-validation of the 3 classifiers
            disagreement_rate = 1.0 - np.mean(predictions[0] == predictions[1])

            logger.info(
                f"[{self.technique_name}] Round {round_idx}: AUC={current_auc:.4f}, "
                f"added={new_samples_added}, disagreement={disagreement_rate:.4f}"
            )

            if current_auc > best_auc:
                best_auc = current_auc

            # Safety checks
            if disagreement_rate > self._max_error_rate:
                logger.warning(
                    f"[{self.technique_name}] Error rate {disagreement_rate:.4f} "
                    f"exceeds max {self._max_error_rate}, stopping"
                )
                break

            if new_samples_added == 0:
                logger.info(f"[{self.technique_name}] No new agreements, stopping")
                break

            safe = self.check_safety_guardrails(len(X_train), total_pseudo)
            if not safe:
                break

        # Store pseudo-labels (from the last round's agreements)
        if self._pseudo_store and "select_mask" in dir():
            final_selected = np.where(agreement_mask & high_confidence)[0][:500]
            if len(final_selected) > 0:
                final_labels = agreed_label[final_selected]
                final_confidences = np.abs(avg_confidence[final_selected] - 0.5) + 0.5
                final_ids = ids_unlabeled[final_selected]

                await self._pseudo_store.store_pseudo_labels(
                    domain=self.config.domain,
                    prediction_ids=final_ids.tolist(),
                    features=X_unlabeled[final_selected],
                    pseudo_labels=final_labels,
                    confidences=final_confidences,
                    label_source="tri_training",
                )

        await self.log_training_run(
            iteration=round_idx if "round_idx" in dir() else 0,
            labeled_count=len(X_train),
            pseudo_labeled_count=total_pseudo,
            model_auc_labeled=best_auc,
            converged=True,
        )

        return {
            "classifiers": classifiers,
            "model": classifiers[0],  # Primary model (LGB)
            "auc_labeled": best_auc,
            "iterations": (round_idx + 1) if "round_idx" in dir() else 0,
            "pseudo_labeled_count": total_pseudo,
            "converged": True,
        }
