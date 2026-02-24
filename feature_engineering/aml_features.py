"""AML Feature Extractor — Retrieves and computes AML risk features from Redis + PostgreSQL."""

import json
from datetime import datetime

import redis.asyncio as redis
from loguru import logger

from serving.app.schemas.aml import AMLFeatureVector, AMLScoreRequest

# High-risk jurisdictions per FATF grey/black list
HIGH_RISK_COUNTRIES = {
    "IR", "KP", "MM", "SY", "YE", "AF",  # FATF blacklist
    "PK", "NG", "AL", "BB", "BF", "CM",  # FATF greylist (subset)
    "CD", "GH", "HT", "JM", "JO", "ML",
    "MZ", "PA", "PH", "SN", "SS", "TZ",
    "TR", "UG", "VN", "ZA",
}

SANCTIONED_COUNTRIES = {"IR", "KP", "SY", "CU", "VE"}


class AMLFeatureExtractor:
    def __init__(self, redis_client: redis.Redis, db_pool: any = None):
        self._redis = redis_client
        self._db = db_pool

    async def extract(self, request: AMLScoreRequest) -> AMLFeatureVector:
        """Extract full feature vector for AML scoring."""
        user_id = str(request.user_id)

        velocity = await self._get_velocity_features(user_id)
        user_features = await self._get_user_aggregate_features(user_id)
        geo_features = await self._compute_geo_features(request)
        behavioral = await self._compute_behavioral_features(user_id, request)
        network = await self._get_network_features(user_id)

        return AMLFeatureVector(
            # Transaction level
            amount=request.amount,
            velocity_1h=velocity.get("txn_count_1h", 0),
            velocity_24h=velocity.get("txn_count_24h", 0),
            total_volume_24h=velocity.get("volume_24h", 0.0),
            avg_amount_30d=user_features.get("avg_transaction_amount", 0.0),
            std_amount_30d=user_features.get("std_transaction_amount", 0.0),
            time_of_day=request.timestamp.hour / 24.0,
            is_cross_border=geo_features["is_cross_border"],
            # User level
            account_age_days=user_features.get("account_age_days", 365),
            device_count_30d=user_features.get("device_count_30d", 1),
            ip_count_30d=user_features.get("ip_count_30d", 1),
            new_device_flag=await self._check_new_device(user_id, request.device_id),
            kyc_completeness_score=user_features.get("kyc_completeness_score", 1.0),
            # Network level
            network_risk_score=network.get("network_risk_score", 0.0),
            circular_transfer_flag=network.get("circular_transfer_flag", 0),
            shared_device_cluster_size=network.get("shared_device_cluster_size", 0),
            # Geographic
            high_risk_country_flag=geo_features["high_risk_country_flag"],
            sanctions_proximity_score=geo_features["sanctions_proximity_score"],
            ip_country_mismatch=geo_features["ip_country_mismatch"],
            # Behavioral
            historical_structuring_flag=behavioral.get("historical_structuring_flag", 0),
            structuring_score_24h=behavioral.get("structuring_score_24h", 0.0),
            rapid_drain_flag=behavioral.get("rapid_drain_flag", 0),
        )

    async def _get_velocity_features(self, user_id: str) -> dict:
        """Real-time velocity from Redis atomic counters."""
        try:
            pipe = self._redis.pipeline()
            pipe.get(f"aml:user:{user_id}:txn_count:1h")
            pipe.get(f"aml:user:{user_id}:txn_count:24h")
            pipe.get(f"aml:user:{user_id}:volume:24h")
            results = await pipe.execute()
            return {
                "txn_count_1h": int(results[0] or 0),
                "txn_count_24h": int(results[1] or 0),
                "volume_24h": float(results[2] or 0.0),
            }
        except Exception as e:
            logger.warning(f"Redis velocity fetch failed: {e}")
            return {"txn_count_1h": 0, "txn_count_24h": 0, "volume_24h": 0.0}

    async def _get_user_aggregate_features(self, user_id: str) -> dict:
        """Persistent user features from Redis cache / PostgreSQL fallback."""
        cache_key = f"aml:user:{user_id}:aggregate_features"
        try:
            cached = await self._redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis aggregate feature fetch failed: {e}")

        if self._db:
            try:
                async with self._db.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM aml_user_features WHERE user_id = $1", user_id
                    )
                    if row:
                        features = dict(row)
                        try:
                            await self._redis.set(cache_key, json.dumps(features, default=str), ex=300)
                        except Exception:
                            pass
                        return features
            except Exception as e:
                logger.warning(f"PostgreSQL aggregate feature fallback failed: {e}")

        return {
            "avg_transaction_amount": 0.0,
            "std_transaction_amount": 0.0,
            "device_count_30d": 1,
            "ip_count_30d": 1,
            "account_age_days": 365,
            "kyc_completeness_score": 1.0,
        }

    async def _get_network_features(self, user_id: str) -> dict:
        """Network/graph risk features from Redis cache."""
        cache_key = f"aml:user:{user_id}:network_features"
        try:
            cached = await self._redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis network feature fetch failed: {e}")

        return {
            "network_risk_score": 0.0,
            "circular_transfer_flag": 0,
            "shared_device_cluster_size": 0,
        }

    async def _check_ip_country_mismatch(self, request: AMLScoreRequest) -> int:
        """Compare IP address geo (from Redis) against declared geo_location country.

        Returns 1 if mismatch detected, 0 otherwise. Falls back to 0 if no data available.
        """
        if not request.ip_address or not request.geo_location:
            return 0

        try:
            ip_country = await self._redis.get(f"ip_geo:{request.ip_address}")
            if ip_country:
                ip_country_str = ip_country.decode() if isinstance(ip_country, bytes) else str(ip_country)
                declared_country = request.geo_location.country.upper()
                if ip_country_str.upper() != declared_country:
                    return 1
        except Exception as e:
            logger.warning(f"IP-country mismatch check failed: {e}")

        return 0

    async def _compute_geo_features(self, request: AMLScoreRequest) -> dict:
        """Compute geographic risk features."""
        country = request.geo_location.country.upper() if request.geo_location else ""

        high_risk_flag = 1 if country in HIGH_RISK_COUNTRIES else 0
        sanctions_proximity = 1.0 if country in SANCTIONED_COUNTRIES else (0.5 if country in HIGH_RISK_COUNTRIES else 0.0)

        # Cross-border: if sender/receiver wallets differ and geo indicates different country
        is_cross_border = 0
        if request.sender_wallet_id and request.receiver_wallet_id:
            if request.sender_wallet_id != request.receiver_wallet_id:
                is_cross_border = 1 if high_risk_flag else 0

        # IP-country mismatch: compare IP geo from Redis vs declared geo
        ip_country_mismatch = await self._check_ip_country_mismatch(request)

        return {
            "high_risk_country_flag": high_risk_flag,
            "sanctions_proximity_score": sanctions_proximity,
            "is_cross_border": is_cross_border,
            "ip_country_mismatch": ip_country_mismatch,
        }

    async def _check_new_device(self, user_id: str, device_id: str | None) -> int:
        """Check if device is new for this user."""
        if not device_id:
            return 0
        try:
            known_devices_key = f"aml:user:{user_id}:known_devices"
            is_member = await self._redis.sismember(known_devices_key, device_id)
            if not is_member:
                await self._redis.sadd(known_devices_key, device_id)
                await self._redis.expire(known_devices_key, 86400 * 30)
                return 1
            return 0
        except Exception:
            return 0

    async def _compute_behavioral_features(self, user_id: str, request: AMLScoreRequest) -> dict:
        """Compute behavioral AML features (structuring, rapid drain)."""
        features: dict = {"historical_structuring_flag": 0, "structuring_score_24h": 0.0, "rapid_drain_flag": 0}

        try:
            # Structuring detection: many small transactions just under threshold
            reporting_threshold = 1_000_000  # KES 1M or equivalent
            txn_count_24h = int(await self._redis.get(f"aml:user:{user_id}:txn_count:24h") or 0)
            volume_24h = float(await self._redis.get(f"aml:user:{user_id}:volume:24h") or 0.0)

            if txn_count_24h >= 5 and request.amount < reporting_threshold and volume_24h + request.amount > reporting_threshold:
                features["structuring_score_24h"] = min(1.0, txn_count_24h / 10.0)

            # Rapid drain: large outflow relative to balance
            if volume_24h > 0 and request.amount > volume_24h * 0.5:
                features["rapid_drain_flag"] = 1

        except Exception as e:
            logger.warning(f"Behavioral feature computation failed: {e}")

        return features

    async def update_velocity_counters(self, user_id: str, amount: float) -> None:
        """Atomically increment AML velocity counters."""
        try:
            pipe = self._redis.pipeline()
            key_1h = f"aml:user:{user_id}:txn_count:1h"
            pipe.incr(key_1h)
            pipe.expire(key_1h, 3600)
            key_24h = f"aml:user:{user_id}:txn_count:24h"
            pipe.incr(key_24h)
            pipe.expire(key_24h, 86400)
            vol_key = f"aml:user:{user_id}:volume:24h"
            pipe.incrbyfloat(vol_key, amount)
            pipe.expire(vol_key, 86400)
            await pipe.execute()
        except Exception as e:
            logger.error(f"Failed to update AML velocity counters: {e}")
