"""Active Learning Service — Intelligent sample selection for human review.

Implements multiple query strategies to select the most informative
predictions for labeling, maximizing model improvement per annotation dollar.

Strategies:
  UNCERTAINTY  — samples nearest decision boundary (min abs distance to thresholds)
  MARGIN       — smallest margin between positive/negative prediction
  ENTROPY      — maximum information-theoretic uncertainty
  DISAGREEMENT — largest gap between ML model and anomaly model scores
  DENSITY_WEIGHTED — uncertainty weighted by local density (representative samples)
"""

import math
import uuid
from datetime import datetime, timedelta
from typing import Any

import redis.asyncio as redis
from loguru import logger

from serving.app.schemas.labels import (
    ALConfigResponse,
    ALConfigUpdateRequest,
    ALMetricsResponse,
    ALQueueItem,
    ALQueueResponse,
    ALQueueStatus,
    ALStrategy,
    Domain,
)


# Decision thresholds per domain for uncertainty calculation
_DOMAIN_THRESHOLDS = {
    Domain.FRAUD: [0.3, 0.7, 0.9],
    Domain.AML: [0.4, 0.7, 0.85],
    Domain.MERCHANT: [0.3, 0.6, 0.85],
}

_DOMAIN_TABLE = {
    Domain.FRAUD: "ml_predictions",
    Domain.AML: "aml_predictions",
    Domain.MERCHANT: "merchant_risk_predictions",
}


class ActiveLearningService:
    """Manages active learning queue, query strategies, and budget tracking."""

    def __init__(
        self,
        db_pool: Any,
        redis_client: redis.Redis,
    ):
        self._db = db_pool
        self._redis = redis_client

    # ─── Query Strategies ──────────────────────────────────────────────

    def _uncertainty_score(self, risk_score: float, domain: Domain) -> float:
        """Uncertainty sampling: 1.0 - min distance to any threshold."""
        thresholds = _DOMAIN_THRESHOLDS[domain]
        min_dist = min(abs(risk_score - t) for t in thresholds)
        return max(0.0, 1.0 - min_dist)

    def _margin_score(self, risk_score: float) -> float:
        """Margin sampling: 1.0 - abs(2*score - 1)."""
        return max(0.0, 1.0 - abs(2.0 * risk_score - 1.0))

    def _entropy_score(self, risk_score: float) -> float:
        """Entropy sampling: -p*log(p) - (1-p)*log(1-p), normalized to [0,1]."""
        p = max(1e-10, min(1.0 - 1e-10, risk_score))
        entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        return entropy  # Max entropy is 1.0 when p=0.5

    def _disagreement_score(self, ml_score: float, anomaly_score: float) -> float:
        """Disagreement sampling: abs(ml_score - anomaly_score)."""
        return abs(ml_score - anomaly_score)

    def _compute_informativeness(
        self,
        risk_score: float,
        domain: Domain,
        strategy: ALStrategy,
        anomaly_score: float | None = None,
    ) -> float:
        """Compute informativeness score using the specified strategy."""
        if strategy == ALStrategy.UNCERTAINTY:
            return self._uncertainty_score(risk_score, domain)
        elif strategy == ALStrategy.MARGIN:
            return self._margin_score(risk_score)
        elif strategy == ALStrategy.ENTROPY:
            return self._entropy_score(risk_score)
        elif strategy == ALStrategy.DISAGREEMENT:
            if anomaly_score is not None:
                return self._disagreement_score(risk_score, anomaly_score)
            return self._uncertainty_score(risk_score, domain)
        elif strategy == ALStrategy.DENSITY_WEIGHTED:
            # Base uncertainty * density proxy (score closer to 0.5 = denser region)
            base = self._uncertainty_score(risk_score, domain)
            density_proxy = 1.0 - abs(risk_score - 0.5) * 2.0
            return base * (0.5 + 0.5 * density_proxy)
        return self._uncertainty_score(risk_score, domain)

    # ─── Queue Management ──────────────────────────────────────────────

    async def refresh_queue(self, domain: Domain) -> int:
        """Recompute active learning queue for a domain.

        Queries unlabeled predictions, scores informativeness, stores top items.
        Returns number of items added.
        """
        if not self._db:
            return 0

        # Get AL config for this domain
        config = await self._get_config(domain)
        if not config or not config.get("is_active"):
            return 0

        strategy = ALStrategy(config["strategy"])
        table = _DOMAIN_TABLE[domain]
        expires_at = datetime.utcnow() + timedelta(hours=48)

        async with self._db.acquire() as conn:
            # Clear expired items
            await conn.execute(
                """
                UPDATE active_learning_queue
                SET status = 'EXPIRED'
                WHERE domain = $1 AND status = 'PENDING' AND expires_at < NOW()
                """,
                domain.value,
            )

            # Get unlabeled predictions not already in queue
            if domain == Domain.FRAUD:
                id_col = "user_id"
            elif domain == Domain.AML:
                id_col = "user_id"
            else:
                id_col = "merchant_id"

            rows = await conn.fetch(
                f"""
                SELECT p.id, p.risk_score, p.{id_col} as entity_id
                FROM {table} p
                LEFT JOIN active_learning_queue alq
                    ON alq.prediction_id = p.id::text
                    AND alq.status IN ('PENDING', 'ASSIGNED')
                WHERE p.label IS NULL
                  AND alq.id IS NULL
                  AND p.created_at > NOW() - INTERVAL '30 days'
                ORDER BY p.created_at DESC
                LIMIT 5000
                """,
            )

            # Score and rank
            scored = []
            for row in rows:
                score = float(row["risk_score"])
                informativeness = self._compute_informativeness(score, domain, strategy)
                scored.append((row, informativeness))

            # Sort by informativeness descending, take top N (daily budget * 2)
            scored.sort(key=lambda x: x[1], reverse=True)
            budget = int(config.get("daily_budget", 100))
            top_items = scored[: budget * 2]

            # Batch insert into queue
            added = 0
            for row, informativeness in top_items:
                try:
                    await conn.execute(
                        """
                        INSERT INTO active_learning_queue
                        (id, domain, prediction_id, entity_id, risk_score, informativeness_score, strategy, expires_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        str(uuid.uuid4()),
                        domain.value,
                        str(row["id"]),
                        str(row["entity_id"]) if row["entity_id"] else None,
                        float(row["risk_score"]),
                        round(informativeness, 4),
                        strategy.value,
                        expires_at,
                    )
                    added += 1
                except Exception as e:
                    logger.debug(f"AL queue insert skipped: {e}")

            # Mirror top items to Redis sorted set for fast retrieval
            redis_key = f"pesaflow:al:queue:{domain.value}"
            try:
                await self._redis.delete(redis_key)
                if top_items:
                    members = {str(row["id"]): informativeness for row, informativeness in top_items[:budget]}
                    if members:
                        await self._redis.zadd(redis_key, members)
                        await self._redis.expire(redis_key, 172800)
            except Exception as e:
                logger.warning(f"Redis AL queue update failed: {e}")

        return added

    async def get_queue(
        self,
        domain: Domain,
        limit: int = 50,
        offset: int = 0,
    ) -> ALQueueResponse:
        """Get prioritized review queue for a domain."""
        config = await self._get_config(domain)
        strategy = ALStrategy(config["strategy"]) if config else ALStrategy.UNCERTAINTY

        items = []
        total = 0

        if self._db:
            async with self._db.acquire() as conn:
                count_row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) as total FROM active_learning_queue
                    WHERE domain = $1 AND status = 'PENDING'
                    """,
                    domain.value,
                )
                total = int(count_row["total"])

                rows = await conn.fetch(
                    """
                    SELECT * FROM active_learning_queue
                    WHERE domain = $1 AND status = 'PENDING'
                    ORDER BY informativeness_score DESC
                    LIMIT $2 OFFSET $3
                    """,
                    domain.value,
                    limit,
                    offset,
                )

                for row in rows:
                    items.append(
                        ALQueueItem(
                            id=row["id"],
                            domain=Domain(row["domain"]),
                            prediction_id=row["prediction_id"],
                            entity_id=row["entity_id"],
                            risk_score=float(row["risk_score"]) if row["risk_score"] else None,
                            informativeness_score=float(row["informativeness_score"]),
                            strategy=ALStrategy(row["strategy"]),
                            status=ALQueueStatus(row["status"]),
                            assigned_to=row.get("assigned_to"),
                            expires_at=row.get("expires_at"),
                            created_at=row["created_at"],
                        )
                    )

        budget_remaining = await self._get_budget_remaining(domain)

        return ALQueueResponse(
            domain=domain,
            total=total,
            items=items,
            strategy=strategy,
            budget_remaining_today=budget_remaining,
        )

    async def create_review_cases(self, domain: Domain, count: int = 10) -> list[str]:
        """Auto-create review cases from top queue items.

        For fraud: creates entries flagged for review.
        For AML: creates aml_cases.
        """
        if not self._db:
            return []

        created_ids = []
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, prediction_id, entity_id, risk_score
                FROM active_learning_queue
                WHERE domain = $1 AND status = 'PENDING'
                ORDER BY informativeness_score DESC
                LIMIT $2
                """,
                domain.value,
                count,
            )

            for row in rows:
                try:
                    if domain == Domain.AML:
                        case_id = str(uuid.uuid4())
                        await conn.execute(
                            """
                            INSERT INTO aml_cases
                            (id, entity_type, entity_id, trigger_reason, risk_score, priority, status, prediction_id)
                            VALUES ($1, 'TRANSACTION', $2, $3, $4, 'HIGH', 'OPEN', $5)
                            """,
                            case_id,
                            str(row["entity_id"]) if row["entity_id"] else str(row["prediction_id"]),
                            "ACTIVE_LEARNING_QUEUE",
                            float(row["risk_score"]) if row["risk_score"] else 0.5,
                            str(row["prediction_id"]),
                        )
                        created_ids.append(row["prediction_id"])

                    # Mark queue item as assigned
                    await conn.execute(
                        "UPDATE active_learning_queue SET status = 'ASSIGNED' WHERE id = $1",
                        row["id"],
                    )
                except Exception as e:
                    logger.error(f"Failed to create review case from AL queue: {e}")

        return created_ids

    # ─── Config Management ─────────────────────────────────────────────

    async def _get_config(self, domain: Domain) -> dict | None:
        """Get active learning config for a domain."""
        if not self._db:
            return None

        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM active_learning_config WHERE domain = $1",
                domain.value,
            )
        return dict(row) if row else None

    async def get_config(self, domain: Domain) -> ALConfigResponse | None:
        """Get active learning config as response schema."""
        config = await self._get_config(domain)
        if not config:
            return None

        return ALConfigResponse(
            domain=domain,
            strategy=ALStrategy(config["strategy"]),
            daily_budget=config["daily_budget"],
            weekly_budget=config["weekly_budget"],
            uncertainty_threshold=float(config["uncertainty_threshold"]),
            is_active=config["is_active"],
            updated_at=config.get("updated_at"),
        )

    async def update_config(self, domain: Domain, update: ALConfigUpdateRequest) -> ALConfigResponse:
        """Update active learning config for a domain."""
        if not self._db:
            raise RuntimeError("Database not available")

        async with self._db.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT * FROM active_learning_config WHERE domain = $1",
                domain.value,
            )
            if not existing:
                raise ValueError(f"No config found for domain {domain.value}")

            new_strategy = update.strategy.value if update.strategy else existing["strategy"]
            new_daily = update.daily_budget if update.daily_budget is not None else existing["daily_budget"]
            new_weekly = update.weekly_budget if update.weekly_budget is not None else existing["weekly_budget"]
            new_threshold = (
                update.uncertainty_threshold
                if update.uncertainty_threshold is not None
                else float(existing["uncertainty_threshold"])
            )
            new_active = update.is_active if update.is_active is not None else existing["is_active"]

            await conn.execute(
                """
                UPDATE active_learning_config
                SET strategy = $2, daily_budget = $3, weekly_budget = $4,
                    uncertainty_threshold = $5, is_active = $6, updated_at = NOW()
                WHERE domain = $1
                """,
                domain.value,
                new_strategy,
                new_daily,
                new_weekly,
                new_threshold,
                new_active,
            )

        return await self.get_config(domain)

    # ─── Budget Tracking ───────────────────────────────────────────────

    async def _get_budget_remaining(self, domain: Domain) -> int:
        """Get remaining daily annotation budget from Redis."""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            used_key = f"pesaflow:al:budget:{domain.value}:daily:{today}"
            used = int(await self._redis.get(used_key) or 0)

            config = await self._get_config(domain)
            daily_budget = config["daily_budget"] if config else 100

            return max(0, daily_budget - used)
        except Exception:
            return 0

    async def record_label_from_queue(self, domain: Domain) -> None:
        """Record that a label was submitted from the AL queue (budget tracking)."""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            daily_key = f"pesaflow:al:budget:{domain.value}:daily:{today}"
            await self._redis.incr(daily_key)
            await self._redis.expire(daily_key, 172800)

            week_start = (datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())).strftime("%Y-%m-%d")
            weekly_key = f"pesaflow:al:budget:{domain.value}:weekly:{week_start}"
            await self._redis.incr(weekly_key)
            await self._redis.expire(weekly_key, 691200)
        except Exception as e:
            logger.warning(f"AL budget tracking failed: {e}")

    # ─── Metrics ───────────────────────────────────────────────────────

    async def get_metrics(self, domain: Domain) -> ALMetricsResponse:
        """Get active learning effectiveness metrics."""
        config = await self._get_config(domain)
        strategy = ALStrategy(config["strategy"]) if config else ALStrategy.UNCERTAINTY
        daily_budget = config["daily_budget"] if config else 100
        weekly_budget = config["weekly_budget"] if config else 500

        total_queued = 0
        total_labeled = 0
        avg_informativeness = 0.0

        if self._db:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'PENDING') as queued,
                        COUNT(*) FILTER (WHERE status = 'LABELED') as labeled,
                        AVG(informativeness_score) FILTER (WHERE status = 'PENDING') as avg_info
                    FROM active_learning_queue
                    WHERE domain = $1
                    """,
                    domain.value,
                )
                if row:
                    total_queued = int(row["queued"] or 0)
                    total_labeled = int(row["labeled"] or 0)
                    avg_informativeness = float(row["avg_info"] or 0)

        # Get daily/weekly counts from Redis
        labels_today = 0
        labels_this_week = 0
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            labels_today = int(await self._redis.get(f"pesaflow:al:budget:{domain.value}:daily:{today}") or 0)
            week_start = (datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())).strftime("%Y-%m-%d")
            labels_this_week = int(await self._redis.get(f"pesaflow:al:budget:{domain.value}:weekly:{week_start}") or 0)
        except Exception:
            pass

        yield_rate = total_labeled / max(1, total_queued + total_labeled)

        return ALMetricsResponse(
            domain=domain,
            total_queued=total_queued,
            total_labeled_from_queue=total_labeled,
            labels_today=labels_today,
            labels_this_week=labels_this_week,
            daily_budget=daily_budget,
            weekly_budget=weekly_budget,
            avg_informativeness=round(avg_informativeness, 4),
            label_yield_rate=round(yield_rate, 4),
            strategy=strategy,
        )

    async def expire_stale_items(self) -> int:
        """Expire queue items past their expiry time."""
        if not self._db:
            return 0

        async with self._db.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE active_learning_queue
                SET status = 'EXPIRED'
                WHERE status = 'PENDING' AND expires_at < NOW()
                """
            )
            # asyncpg returns "UPDATE N" string
            count = int(result.split()[-1]) if result else 0
            return count
