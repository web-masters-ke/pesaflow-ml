"""Fraud Feature Extractor — Retrieves and computes fraud detection features from Redis + PostgreSQL."""

import json
import math
from datetime import datetime

import redis.asyncio as redis
from loguru import logger

from serving.app.schemas.fraud import FraudFeatureVector, FraudScoreRequest

# High-risk currencies (volatile/sanction-exposed)
HIGH_RISK_CURRENCIES = {"IRR", "KPW", "SYP", "VEF", "ZWD", "SDG", "MMK", "CUP"}
MEDIUM_RISK_CURRENCIES = {"NGN", "GHS", "TZS", "UGX", "ETB"}


class FraudFeatureExtractor:
    def __init__(self, redis_client: redis.Redis, db_pool: any = None):
        self._redis = redis_client
        self._db = db_pool

    async def extract(self, request: FraudScoreRequest) -> FraudFeatureVector:
        """Extract full feature vector for fraud scoring."""
        user_features = await self._get_user_features(str(request.user_id))
        device_features = await self._get_device_features(request.device_fingerprint)
        tx_features = await self._compute_transaction_features(request)

        return FraudFeatureVector(
            # User-based
            avg_transaction_amount_7d=user_features.get("avg_transaction_amount_7d", 0.0),
            transaction_velocity_1h=user_features.get("transaction_velocity_1h", 0),
            transaction_velocity_24h=user_features.get("transaction_velocity_24h", 0),
            failed_login_attempts_24h=user_features.get("failed_login_attempts_24h", 0),
            account_age_days=user_features.get("account_age_days", 0),
            historical_fraud_flag=user_features.get("historical_fraud_flag", 0),
            # Device-based
            device_risk_score=device_features.get("device_risk_score", 0.0),
            device_fraud_count=device_features.get("device_fraud_count", 0),
            distinct_user_count=device_features.get("distinct_user_count", 0),
            # Transaction-based
            amount=request.amount,
            geo_distance_from_last_tx=tx_features["geo_distance"],
            time_of_day=tx_features["time_of_day"],
            currency_risk=tx_features["currency_risk"],
        )

    async def _get_user_features(self, user_id: str) -> dict:
        """Fetch user features: Redis first, PostgreSQL fallback."""
        cache_key = f"user:{user_id}:features"
        try:
            cached = await self._redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis user feature lookup failed: {e}")

        # PostgreSQL fallback
        if self._db:
            try:
                async with self._db.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM feature_store_user WHERE user_id = $1", user_id
                    )
                    if row:
                        features = dict(row)
                        # Cache for next time
                        try:
                            await self._redis.set(cache_key, json.dumps(features, default=str), ex=300)
                        except Exception:
                            pass
                        return features
            except Exception as e:
                logger.warning(f"PostgreSQL user feature fallback failed: {e}")

        # Baseline defaults (conservative)
        logger.warning(f"Using baseline features for user {user_id}")
        return {
            "avg_transaction_amount_7d": 0.0,
            "transaction_velocity_1h": 0,
            "transaction_velocity_24h": 0,
            "failed_login_attempts_24h": 0,
            "account_age_days": 365,
            "historical_fraud_flag": 0,
        }

    async def _get_device_features(self, device_fingerprint: str | None) -> dict:
        """Fetch device features from Redis."""
        if not device_fingerprint:
            return {"device_risk_score": 0.5, "device_fraud_count": 0, "distinct_user_count": 1}

        cache_key = f"device:{device_fingerprint}:features"
        try:
            cached = await self._redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis device feature lookup failed: {e}")

        return {"device_risk_score": 0.1, "device_fraud_count": 0, "distinct_user_count": 1}

    async def _get_last_known_location(self, user_id: str) -> tuple[float, float] | None:
        """Fetch last known transaction geo location from Redis."""
        try:
            cached = await self._redis.get(f"user:{user_id}:last_geo")
            if cached:
                data = json.loads(cached)
                return (float(data["lat"]), float(data["lng"]))
        except Exception as e:
            logger.warning(f"Redis last-geo lookup failed: {e}")
        return None

    async def _store_last_known_location(self, user_id: str, lat: float, lng: float) -> None:
        """Store current geo as last known location with 24h TTL."""
        try:
            data = json.dumps({"lat": lat, "lng": lng})
            await self._redis.set(f"user:{user_id}:last_geo", data, ex=86400)
        except Exception as e:
            logger.warning(f"Failed to store last-geo: {e}")

    @staticmethod
    def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Compute Haversine distance in kilometers between two points."""
        R = 6371.0  # Earth radius in km
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    async def _compute_transaction_features(self, request: FraudScoreRequest) -> dict:
        """Compute derived transaction features from request payload."""
        # Time of day (normalized 0-1, where night hours = higher risk)
        hour = request.timestamp.hour
        time_of_day = hour / 24.0

        # Currency risk scoring
        currency = request.currency.upper()
        if currency in HIGH_RISK_CURRENCIES:
            currency_risk = 0.9
        elif currency in MEDIUM_RISK_CURRENCIES:
            currency_risk = 0.4
        else:
            currency_risk = 0.1

        # Geo distance from last known transaction location
        geo_distance = 0.0
        user_id = str(request.user_id)
        if request.geo_location:
            last_loc = await self._get_last_known_location(user_id)
            if last_loc:
                geo_distance = self._haversine_distance(
                    request.geo_location.lat, request.geo_location.lng,
                    last_loc[0], last_loc[1],
                )
            # Store current location for next transaction
            await self._store_last_known_location(
                user_id, request.geo_location.lat, request.geo_location.lng
            )

        return {
            "time_of_day": time_of_day,
            "currency_risk": currency_risk,
            "geo_distance": geo_distance,
        }

    async def update_velocity_counters(self, user_id: str, amount: float, geo_location: any = None) -> None:
        """Atomically increment velocity counters after scoring."""
        try:
            pipe = self._redis.pipeline()
            # 1-hour velocity
            key_1h = f"user:{user_id}:txn_count:1h"
            pipe.incr(key_1h)
            pipe.expire(key_1h, 3600)
            # 24-hour velocity
            key_24h = f"user:{user_id}:txn_count:24h"
            pipe.incr(key_24h)
            pipe.expire(key_24h, 86400)
            # Volume counters
            vol_key_24h = f"user:{user_id}:volume:24h"
            pipe.incrbyfloat(vol_key_24h, amount)
            pipe.expire(vol_key_24h, 86400)
            await pipe.execute()
        except Exception as e:
            logger.error(f"Failed to update velocity counters: {e}")
