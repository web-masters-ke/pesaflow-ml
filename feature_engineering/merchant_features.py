"""Merchant Feature Extractor — Computes merchant risk features from Redis + PostgreSQL."""

import json
import time
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from loguru import logger

from serving.app.schemas.merchant import MerchantFeatureVector, MerchantScoreRequest


class MerchantFeatureExtractor:
    """Extracts merchant risk features from real-time and historical sources."""

    def __init__(self, redis_client: redis.Redis, db_pool: Any = None):
        self._redis = redis_client
        self._db = db_pool

    async def extract(self, request: MerchantScoreRequest) -> MerchantFeatureVector:
        """Extract all merchant features for scoring."""
        merchant_id = str(request.merchant_id)

        # Fetch from Redis cache first
        cache_key = f"merchant:features:{merchant_id}"
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                return MerchantFeatureVector(**data)
            except Exception:
                pass

        # Extract features from multiple sources
        velocity = await self._get_velocity_features(merchant_id)
        aggregate = await self._get_aggregate_features(merchant_id)
        risk_signals = await self._get_risk_signals(merchant_id)

        features = MerchantFeatureVector(
            # Velocity features
            transaction_count_1h=velocity.get("txn_count_1h", 0),
            transaction_count_24h=velocity.get("txn_count_24h", 0),
            transaction_volume_24h=velocity.get("txn_volume_24h", 0.0),
            unique_customers_24h=velocity.get("unique_customers_24h", 0),
            # Aggregate features
            avg_transaction_amount_30d=aggregate.get("avg_amount_30d", 0.0),
            std_transaction_amount_30d=aggregate.get("std_amount_30d", 0.0),
            chargeback_rate_90d=aggregate.get("chargeback_rate_90d", 0.0),
            refund_rate_90d=aggregate.get("refund_rate_90d", 0.0),
            account_age_days=aggregate.get("account_age_days", 0),
            # Risk signals
            fraud_transaction_rate=risk_signals.get("fraud_txn_rate", 0.0),
            high_risk_customer_ratio=risk_signals.get("high_risk_customer_ratio", 0.0),
            cross_border_ratio=risk_signals.get("cross_border_ratio", 0.0),
            velocity_spike_flag=risk_signals.get("velocity_spike_flag", 0),
            mcc_risk_score=risk_signals.get("mcc_risk_score", 0.0),
            avg_customer_risk_score=risk_signals.get("avg_customer_risk_score", 0.0),
        )

        # Cache features
        try:
            await self._redis.setex(cache_key, 300, json.dumps(features.model_dump()))
        except Exception:
            pass

        return features

    async def _get_velocity_features(self, merchant_id: str) -> dict:
        """Get real-time velocity counters from Redis."""
        try:
            pipe = self._redis.pipeline()
            pipe.get(f"merchant:{merchant_id}:txn_count:1h")
            pipe.get(f"merchant:{merchant_id}:txn_count:24h")
            pipe.get(f"merchant:{merchant_id}:txn_volume:24h")
            pipe.scard(f"merchant:{merchant_id}:customers:24h")
            results = await pipe.execute()

            return {
                "txn_count_1h": int(results[0] or 0),
                "txn_count_24h": int(results[1] or 0),
                "txn_volume_24h": float(results[2] or 0.0),
                "unique_customers_24h": int(results[3] or 0),
            }
        except Exception as e:
            logger.warning(f"Redis velocity fetch failed for merchant {merchant_id}: {e}")
            return {}

    async def _get_aggregate_features(self, merchant_id: str) -> dict:
        """Get historical aggregate features from PostgreSQL."""
        if not self._db:
            return {}

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COALESCE(avg_transaction_amount_30d, 0) as avg_amount_30d,
                        COALESCE(std_transaction_amount_30d, 0) as std_amount_30d,
                        COALESCE(chargeback_rate_90d, 0) as chargeback_rate_90d,
                        COALESCE(refund_rate_90d, 0) as refund_rate_90d,
                        COALESCE(EXTRACT(DAY FROM NOW() - created_at)::int, 0) as account_age_days
                    FROM merchant_risk_profiles
                    WHERE merchant_id = $1
                    """,
                    merchant_id,
                )
                return dict(row) if row else {}
        except Exception as e:
            logger.warning(f"DB aggregate fetch failed for merchant {merchant_id}: {e}")
            return {}

    async def _get_risk_signals(self, merchant_id: str) -> dict:
        """Get risk signal features from Redis and PostgreSQL."""
        signals = {}

        # Redis-based signals
        try:
            pipe = self._redis.pipeline()
            pipe.get(f"merchant:{merchant_id}:fraud_txn_rate")
            pipe.get(f"merchant:{merchant_id}:high_risk_customer_ratio")
            pipe.get(f"merchant:{merchant_id}:cross_border_ratio")
            pipe.get(f"merchant:{merchant_id}:velocity_spike")
            pipe.get(f"merchant:{merchant_id}:mcc_risk_score")
            pipe.get(f"merchant:{merchant_id}:avg_customer_risk")
            results = await pipe.execute()

            signals = {
                "fraud_txn_rate": float(results[0] or 0.0),
                "high_risk_customer_ratio": float(results[1] or 0.0),
                "cross_border_ratio": float(results[2] or 0.0),
                "velocity_spike_flag": int(results[3] or 0),
                "mcc_risk_score": float(results[4] or 0.0),
                "avg_customer_risk_score": float(results[5] or 0.0),
            }
        except Exception as e:
            logger.warning(f"Redis risk signals fetch failed for merchant {merchant_id}: {e}")

        return signals

    async def update_velocity_counters(self, merchant_id: str, amount: float, customer_id: str) -> None:
        """Update real-time velocity counters for a merchant."""
        try:
            pipe = self._redis.pipeline()

            # Transaction count (1h window)
            key_1h = f"merchant:{merchant_id}:txn_count:1h"
            pipe.incr(key_1h)
            pipe.expire(key_1h, 3600)

            # Transaction count (24h window)
            key_24h = f"merchant:{merchant_id}:txn_count:24h"
            pipe.incr(key_24h)
            pipe.expire(key_24h, 86400)

            # Transaction volume (24h)
            vol_key = f"merchant:{merchant_id}:txn_volume:24h"
            pipe.incrbyfloat(vol_key, amount)
            pipe.expire(vol_key, 86400)

            # Unique customers (24h)
            cust_key = f"merchant:{merchant_id}:customers:24h"
            pipe.sadd(cust_key, customer_id)
            pipe.expire(cust_key, 86400)

            await pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to update merchant velocity counters: {e}")
