"""SSL Metrics — Pseudo-label quality tracking.

Monitors:
  - Pseudo-label flip rate between iterations
  - Validated pseudo-label accuracy
  - SSL training run performance
  - Cross-domain transfer statistics
"""

from typing import Any

from loguru import logger


class SSLMetricsCollector:
    """Collects and reports SSL quality metrics."""

    def __init__(self, db_pool: Any = None):
        self._db = db_pool

    async def get_pseudo_label_quality(self, domain: str) -> dict:
        """Get pseudo-label quality metrics for a domain."""
        if not self._db:
            return {}

        async with self._db.acquire() as conn:
            # Pseudo-label stats
            pl_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_pseudo_labels,
                    COUNT(*) FILTER (WHERE is_validated) as validated_count,
                    COUNT(*) FILTER (WHERE label_correct = TRUE) as correct_count,
                    COUNT(*) FILTER (WHERE label_correct = FALSE) as incorrect_count,
                    AVG(pseudo_label_confidence) as avg_confidence,
                    COUNT(DISTINCT label_source) as technique_count
                FROM ssl_pseudo_labels
                WHERE domain = $1
                """,
                domain,
            )

            # Latest training run stats
            latest_run = await conn.fetchrow(
                """
                SELECT technique, iteration, labeled_count, pseudo_labeled_count,
                       pseudo_label_agreement_rate, pseudo_label_flip_rate,
                       model_auc_labeled, model_auc_full, converged, created_at
                FROM ssl_training_runs
                WHERE domain = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                domain,
            )

            # Flip rate trend (last 10 runs)
            flip_trend = await conn.fetch(
                """
                SELECT technique, pseudo_label_flip_rate, created_at
                FROM ssl_training_runs
                WHERE domain = $1 AND pseudo_label_flip_rate IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 10
                """,
                domain,
            )

        validated = int(pl_stats["validated_count"] or 0) if pl_stats else 0
        correct = int(pl_stats["correct_count"] or 0) if pl_stats else 0

        result = {
            "domain": domain,
            "total_pseudo_labels": int(pl_stats["total_pseudo_labels"] or 0) if pl_stats else 0,
            "validated_count": validated,
            "validated_accuracy": correct / validated if validated > 0 else None,
            "avg_confidence": float(pl_stats["avg_confidence"] or 0) if pl_stats else 0,
            "technique_count": int(pl_stats["technique_count"] or 0) if pl_stats else 0,
        }

        if latest_run:
            result["latest_run"] = {
                "technique": latest_run["technique"],
                "iteration": latest_run["iteration"],
                "labeled_count": latest_run["labeled_count"],
                "pseudo_labeled_count": latest_run["pseudo_labeled_count"],
                "flip_rate": (
                    float(latest_run["pseudo_label_flip_rate"]) if latest_run["pseudo_label_flip_rate"] else None
                ),
                "model_auc": float(latest_run["model_auc_labeled"]) if latest_run["model_auc_labeled"] else None,
                "converged": latest_run["converged"],
                "created_at": latest_run["created_at"].isoformat() if latest_run["created_at"] else None,
            }

        if flip_trend:
            result["flip_rate_trend"] = [
                {
                    "technique": row["technique"],
                    "flip_rate": float(row["pseudo_label_flip_rate"]) if row["pseudo_label_flip_rate"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in flip_trend
            ]

        return result

    async def get_cross_domain_stats(self) -> dict:
        """Get cross-domain label transfer statistics."""
        if not self._db:
            return {}

        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    source_domain,
                    target_domain,
                    COUNT(*) as transfer_count,
                    AVG(confidence_decay) as avg_decay,
                    MAX(created_at) as last_transfer
                FROM ssl_cross_domain_transfers
                GROUP BY source_domain, target_domain
                ORDER BY transfer_count DESC
                """
            )

        transfers = []
        for row in rows:
            transfers.append(
                {
                    "source_domain": row["source_domain"],
                    "target_domain": row["target_domain"],
                    "transfer_count": int(row["transfer_count"]),
                    "avg_confidence_decay": round(float(row["avg_decay"] or 0), 4),
                    "last_transfer": row["last_transfer"].isoformat() if row["last_transfer"] else None,
                }
            )

        return {"transfers": transfers, "total_transfers": sum(t["transfer_count"] for t in transfers)}

    async def get_technique_comparison(self, domain: str) -> list[dict]:
        """Compare SSL techniques for a domain by their latest AUC."""
        if not self._db:
            return []

        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (technique)
                    technique,
                    model_auc_labeled,
                    model_auc_full,
                    pseudo_labeled_count,
                    pseudo_label_flip_rate,
                    converged,
                    created_at
                FROM ssl_training_runs
                WHERE domain = $1
                ORDER BY technique, created_at DESC
                """,
                domain,
            )

        return [
            {
                "technique": row["technique"],
                "auc_labeled": float(row["model_auc_labeled"]) if row["model_auc_labeled"] else None,
                "auc_full": float(row["model_auc_full"]) if row["model_auc_full"] else None,
                "pseudo_labeled_count": row["pseudo_labeled_count"],
                "flip_rate": float(row["pseudo_label_flip_rate"]) if row["pseudo_label_flip_rate"] else None,
                "converged": row["converged"],
                "last_run": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]

    async def check_alerts(self, domain: str) -> list[dict]:
        """Check for SSL quality alerts."""
        alerts = []
        quality = await self.get_pseudo_label_quality(domain)

        # Check validated accuracy
        accuracy = quality.get("validated_accuracy")
        if accuracy is not None and accuracy < 0.85:
            alerts.append(
                {
                    "severity": "WARNING",
                    "message": f"Pseudo-label accuracy for {domain} is {accuracy:.2%}, below 85% floor",
                    "metric": "pseudo_label_accuracy",
                    "value": accuracy,
                }
            )

        # Check flip rate
        latest = quality.get("latest_run", {})
        flip_rate = latest.get("flip_rate")
        if flip_rate is not None and flip_rate > 0.10:
            alerts.append(
                {
                    "severity": "WARNING",
                    "message": f"Label flip rate for {domain} is {flip_rate:.2%}, exceeding 10% ceiling",
                    "metric": "label_flip_rate",
                    "value": flip_rate,
                }
            )

        return alerts
