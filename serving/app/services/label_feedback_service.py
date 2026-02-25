"""Label Feedback Service — Manages label submission, propagation, and audit.

Labels flow in from:
  - Manual review (analyst clicks approve/reject)
  - Chargeback feeds (card network chargebacks)
  - SAR confirmations (filed SARs confirm suspicious activity)
  - Partner feedback (bank/PSP partner labels)
  - Automated rules (high-confidence rule-based labels)
  - SSL techniques (self-training, co-training, label propagation)

Labels update the prediction tables so DataMaturityDetector can assess maturity
and the system can auto-escalate from COLD → WARMING → WARM → HOT.
"""

import uuid
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from loguru import logger

from serving.app.schemas.labels import (
    Domain,
    DomainLabelStats,
    LabelPropagationResponse,
    LabelSource,
    LabelStatisticsResponse,
    LabelSubmitResponse,
)

# Mapping from prediction domain to DB table
_DOMAIN_TABLE = {
    Domain.FRAUD: "ml_predictions",
    Domain.AML: "aml_predictions",
    Domain.MERCHANT: "merchant_risk_predictions",
}


class LabelFeedbackService:
    """Manages label submission, case-to-label propagation, and audit."""

    def __init__(
        self,
        db_pool: Any,
        redis_client: redis.Redis,
        kafka_publish: Any = None,
    ):
        self._db = db_pool
        self._redis = redis_client
        self._kafka_publish = kafka_publish

    async def submit_label(
        self,
        prediction_id: str,
        domain: Domain,
        label: int,
        label_source: LabelSource,
        labeled_by: str | None = None,
        reason: str | None = None,
    ) -> LabelSubmitResponse:
        """Submit a single label for a prediction."""
        table = _DOMAIN_TABLE[domain]
        now = datetime.utcnow()

        async with self._db.acquire() as conn:
            # Get current label (for audit)
            row = await conn.fetchrow(
                f"SELECT id, label FROM {table} WHERE id = $1",
                prediction_id,
            )
            if not row:
                raise ValueError(f"Prediction {prediction_id} not found in {table}")

            previous_label = row["label"]

            # Update prediction with label
            await conn.execute(
                f"""
                UPDATE {table}
                SET label = $2, labeled_at = $3, labeled_by = $4, label_source = $5
                WHERE id = $1
                """,
                prediction_id,
                label,
                now,
                labeled_by,
                label_source.value,
            )

            # Insert audit record
            await conn.execute(
                """
                INSERT INTO label_audit_history
                (id, domain, prediction_id, previous_label, new_label, label_source, labeled_by, reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                str(uuid.uuid4()),
                domain.value,
                prediction_id,
                previous_label,
                label,
                label_source.value,
                labeled_by,
                reason,
            )

        # Increment Redis label counter
        try:
            counter_key = f"pesaflow:labels:{domain.value}:count"
            await self._redis.incr(counter_key)
            daily_key = f"pesaflow:labels:{domain.value}:daily:{now.strftime('%Y-%m-%d')}"
            await self._redis.incr(daily_key)
            await self._redis.expire(daily_key, 172800)  # 2 days TTL
        except Exception as e:
            logger.warning(f"Redis label counter update failed: {e}")

        # Publish Kafka event
        if self._kafka_publish:
            try:
                await self._kafka_publish(
                    domain.value,
                    prediction_id,
                    {
                        "prediction_id": prediction_id,
                        "domain": domain.value,
                        "label": label,
                        "label_source": label_source.value,
                        "labeled_at": now.isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"Kafka label publish failed: {e}")

        return LabelSubmitResponse(
            prediction_id=uuid.UUID(prediction_id) if isinstance(prediction_id, str) else prediction_id,
            domain=domain,
            label=label,
            label_source=label_source,
            labeled_at=now,
            previous_label=previous_label,
        )

    async def submit_batch(
        self,
        labels: list[dict],
    ) -> dict:
        """Submit a batch of labels, collecting errors."""
        results = []
        errors = []

        for item in labels:
            try:
                result = await self.submit_label(
                    prediction_id=str(item["prediction_id"]),
                    domain=item["domain"],
                    label=item["label"],
                    label_source=item["label_source"],
                    labeled_by=str(item.get("labeled_by")) if item.get("labeled_by") else None,
                    reason=item.get("reason"),
                )
                results.append(result)
            except Exception as e:
                errors.append(
                    {
                        "prediction_id": str(item.get("prediction_id")),
                        "error": str(e),
                    }
                )

        return {
            "total": len(labels),
            "succeeded": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }

    async def propagate_aml_case_labels(self) -> int:
        """Propagate labels from closed AML cases to aml_predictions.

        Maps: CLOSED_CONFIRMED → 1, CLOSED_FALSE_POSITIVE → 0
        """
        if not self._db:
            return 0

        propagated = 0
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, entity_id, status, prediction_id
                FROM aml_cases
                WHERE status LIKE 'CLOSED_%'
                  AND label_propagated = FALSE
                  AND status IN ('CLOSED_CONFIRMED', 'CLOSED_FALSE_POSITIVE')
                """
            )

            for row in rows:
                label = 1 if row["status"] == "CLOSED_CONFIRMED" else 0
                prediction_id = row.get("prediction_id")

                if prediction_id:
                    target_id = str(prediction_id)
                else:
                    # Fallback: find prediction by entity_id (transaction_id)
                    pred = await conn.fetchrow(
                        """
                        SELECT id FROM aml_predictions
                        WHERE transaction_id = $1
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        row["entity_id"],
                    )
                    if not pred:
                        continue
                    target_id = str(pred["id"])

                try:
                    await self.submit_label(
                        prediction_id=target_id,
                        domain=Domain.AML,
                        label=label,
                        label_source=LabelSource.SAR_CONFIRMED if label == 1 else LabelSource.MANUAL_REVIEW,
                        reason=f"Propagated from AML case {row['id']}",
                    )
                    # Mark case as propagated
                    await conn.execute(
                        "UPDATE aml_cases SET label_propagated = TRUE WHERE id = $1",
                        row["id"],
                    )
                    propagated += 1
                except Exception as e:
                    logger.error(f"Failed to propagate label from AML case {row['id']}: {e}")

        return propagated

    async def propagate_fraud_review_labels(self) -> int:
        """Propagate labels from fraud review decisions to ml_predictions.

        Looks for predictions with decision=REVIEW that have been
        subsequently resolved through the review workflow.
        """
        if not self._db:
            return 0

        propagated = 0
        async with self._db.acquire() as conn:
            # Look for reviewed predictions that don't have labels yet
            rows = await conn.fetch(
                """
                SELECT p.id, p.risk_score, p.decision
                FROM ml_predictions p
                WHERE p.decision = 'REVIEW'
                  AND p.label IS NULL
                  AND p.created_at < NOW() - INTERVAL '24 hours'
                LIMIT 500
                """
            )

            # For REVIEW decisions that aged out without explicit review,
            # if score was very high (>0.9) label as positive, very low (<0.1) as negative
            for row in rows:
                score = float(row["risk_score"])
                if score > 0.9:
                    label = 1
                elif score < 0.1:
                    label = 0
                else:
                    continue  # Skip ambiguous cases

                try:
                    await self.submit_label(
                        prediction_id=str(row["id"]),
                        domain=Domain.FRAUD,
                        label=label,
                        label_source=LabelSource.AUTOMATED_RULE,
                        reason=f"Auto-labeled from aged REVIEW (score={score:.4f})",
                    )
                    propagated += 1
                except Exception as e:
                    logger.error(f"Failed to auto-label prediction {row['id']}: {e}")

        return propagated

    async def get_label_statistics(self) -> LabelStatisticsResponse:
        """Get label statistics per domain from materialized view."""
        if not self._db:
            return LabelStatisticsResponse(domains=[], total_labeled=0, total_unlabeled=0)

        now = datetime.utcnow()

        async with self._db.acquire() as conn:
            # Refresh materialized view
            try:
                await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY label_statistics")
            except Exception:
                # CONCURRENTLY requires unique index; fall back to regular refresh
                try:
                    await conn.execute("REFRESH MATERIALIZED VIEW label_statistics")
                except Exception as e:
                    logger.warning(f"Could not refresh label_statistics view: {e}")

            rows = await conn.fetch("SELECT * FROM label_statistics")

        domains = []
        total_labeled = 0
        total_unlabeled = 0

        for row in rows:
            labeled = int(row["labeled_count"] or 0)
            unlabeled = int(row["unlabeled_count"] or 0)

            # Determine maturity level from label count
            if labeled < 100:
                maturity = "COLD"
            elif labeled < 1000:
                maturity = "WARMING"
            elif labeled < 10000:
                maturity = "WARM"
            else:
                maturity = "HOT"

            domains.append(
                DomainLabelStats(
                    domain=Domain(row["domain"]),
                    total_predictions=int(row["total_predictions"] or 0),
                    labeled_count=labeled,
                    positive_count=int(row["positive_count"] or 0),
                    negative_count=int(row["negative_count"] or 0),
                    unlabeled_count=unlabeled,
                    label_rate=float(row["label_rate"] or 0),
                    maturity_level=maturity,
                )
            )
            total_labeled += labeled
            total_unlabeled += unlabeled

        return LabelStatisticsResponse(
            domains=domains,
            total_labeled=total_labeled,
            total_unlabeled=total_unlabeled,
            last_refreshed=now,
        )
