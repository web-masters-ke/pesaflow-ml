"""Pseudo-label store — DB interface for CRUD on ssl_pseudo_labels table."""

import uuid
from typing import Any

import numpy as np
from loguru import logger


class PseudoLabelStore:
    """Manages pseudo-label persistence in ssl_pseudo_labels table."""

    def __init__(self, db_pool: Any):
        self._db = db_pool

    async def store_pseudo_labels(
        self,
        domain: str,
        prediction_ids: list[str],
        features: np.ndarray,
        pseudo_labels: np.ndarray,
        confidences: np.ndarray,
        label_source: str,
        model_version: str | None = None,
        iteration: int = 0,
    ) -> int:
        """Bulk-insert pseudo-labels into ssl_pseudo_labels table."""
        if not self._db:
            return 0

        stored = 0
        async with self._db.acquire() as conn:
            for i, pred_id in enumerate(prediction_ids):
                try:
                    feature_dict = {f"f_{j}": float(features[i][j]) for j in range(features.shape[1])}
                    await conn.execute(
                        """
                        INSERT INTO ssl_pseudo_labels
                        (id, domain, prediction_id, feature_snapshot, pseudo_label,
                         pseudo_label_confidence, label_source, source_model_version, source_iteration)
                        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                        ON CONFLICT DO NOTHING
                        """,
                        str(uuid.uuid4()),
                        domain,
                        pred_id,
                        feature_dict,
                        int(pseudo_labels[i]),
                        round(float(confidences[i]), 4),
                        label_source,
                        model_version,
                        iteration,
                    )
                    stored += 1
                except Exception as e:
                    logger.debug(f"Pseudo-label store skipped: {e}")

        logger.info(f"Stored {stored} pseudo-labels for {domain} ({label_source})")
        return stored

    async def get_pseudo_labels(
        self,
        domain: str,
        label_source: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve pseudo-labels. Returns (features, labels, confidences)."""
        if not self._db:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        conditions = ["domain = $1", "pseudo_label_confidence >= $2"]
        params = [domain, min_confidence]
        param_idx = 3

        if label_source:
            conditions.append(f"label_source = ${param_idx}")
            params.append(label_source)
            param_idx += 1

        where = " AND ".join(conditions)

        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT feature_snapshot, pseudo_label, pseudo_label_confidence
                FROM ssl_pseudo_labels
                WHERE {where}
                ORDER BY pseudo_label_confidence DESC
                LIMIT ${param_idx}
                """,
                *params,
                limit,
            )

        if not rows:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        features = []
        labels = []
        confidences = []
        for row in rows:
            snapshot = row["feature_snapshot"]
            if isinstance(snapshot, dict):
                features.append(list(snapshot.values()))
            elif isinstance(snapshot, list):
                features.append(snapshot)
            else:
                continue
            labels.append(row["pseudo_label"])
            confidences.append(float(row["pseudo_label_confidence"]))

        return np.array(features), np.array(labels), np.array(confidences)

    async def validate_pseudo_labels(
        self,
        domain: str,
        prediction_ids: list[str],
        human_labels: list[int],
    ) -> dict:
        """Validate pseudo-labels against human labels. Updates is_validated, human_label, label_correct."""
        if not self._db:
            return {"validated": 0, "correct": 0, "incorrect": 0}

        correct = 0
        incorrect = 0
        async with self._db.acquire() as conn:
            for pred_id, human_label in zip(prediction_ids, human_labels):
                row = await conn.fetchrow(
                    "SELECT pseudo_label FROM ssl_pseudo_labels WHERE prediction_id = $1 AND domain = $2",
                    pred_id,
                    domain,
                )
                if not row:
                    continue

                is_correct = row["pseudo_label"] == human_label
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1

                await conn.execute(
                    """
                    UPDATE ssl_pseudo_labels
                    SET is_validated = TRUE, human_label = $3, label_correct = $4
                    WHERE prediction_id = $1 AND domain = $2
                    """,
                    pred_id,
                    domain,
                    human_label,
                    is_correct,
                )

        total = correct + incorrect
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Validated {total} pseudo-labels for {domain}: accuracy={accuracy:.4f}")

        return {"validated": total, "correct": correct, "incorrect": incorrect, "accuracy": accuracy}

    async def get_validation_stats(self, domain: str) -> dict:
        """Get pseudo-label validation statistics."""
        if not self._db:
            return {}

        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE is_validated) as validated,
                    COUNT(*) FILTER (WHERE label_correct = TRUE) as correct,
                    COUNT(*) FILTER (WHERE label_correct = FALSE) as incorrect,
                    AVG(pseudo_label_confidence) as avg_confidence
                FROM ssl_pseudo_labels
                WHERE domain = $1
                """,
                domain,
            )

        if not row:
            return {}

        validated = int(row["validated"] or 0)
        correct = int(row["correct"] or 0)

        return {
            "total_pseudo_labels": int(row["total"] or 0),
            "validated": validated,
            "correct": correct,
            "incorrect": int(row["incorrect"] or 0),
            "accuracy": correct / validated if validated > 0 else None,
            "avg_confidence": float(row["avg_confidence"] or 0),
        }
