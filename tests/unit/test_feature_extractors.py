"""Unit tests for Feature Extractors (Fraud + AML)."""

import json
import math
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from serving.app.schemas.aml import AMLScoreRequest
from serving.app.schemas.common import GeoLocation
from serving.app.schemas.fraud import FraudScoreRequest


def _make_fraud_request(**kwargs):
    defaults = {
        "transaction_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "amount": 5000.0,
        "currency": "KES",
        "transaction_type": "WALLET_TRANSFER",
        "timestamp": datetime(2024, 6, 15, 14, 30),
    }
    defaults.update(kwargs)
    return FraudScoreRequest(**defaults)


def _make_aml_request(**kwargs):
    defaults = {
        "transaction_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "amount": 50000.0,
        "currency": "KES",
        "timestamp": datetime(2024, 6, 15, 14, 30),
    }
    defaults.update(kwargs)
    return AMLScoreRequest(**defaults)


class TestFraudFeatureExtractor:
    @pytest.mark.asyncio
    async def test_extract_returns_13_features(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)
        extractor = FraudFeatureExtractor(redis_client=redis_client, db_pool=None)

        req = _make_fraud_request()
        fv = await extractor.extract(req)
        arr = fv.to_array()
        assert len(arr) == 13
        assert fv.amount == 5000.0

    @pytest.mark.asyncio
    async def test_extract_with_cached_user_features(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        cached_data = json.dumps({
            "avg_transaction_amount_7d": 3000.0,
            "transaction_velocity_1h": 5,
            "transaction_velocity_24h": 20,
            "failed_login_attempts_24h": 0,
            "account_age_days": 180,
            "historical_fraud_flag": 0,
        })
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=cached_data)
        extractor = FraudFeatureExtractor(redis_client=redis_client)

        req = _make_fraud_request()
        fv = await extractor.extract(req)
        assert fv.avg_transaction_amount_7d == 3000.0
        assert fv.transaction_velocity_1h == 5

    @pytest.mark.asyncio
    async def test_fallback_to_defaults_on_redis_failure(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        redis_client = AsyncMock()
        redis_client.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        extractor = FraudFeatureExtractor(redis_client=redis_client, db_pool=None)

        req = _make_fraud_request()
        fv = await extractor.extract(req)
        # Should use defaults without raising
        assert fv.account_age_days == 365
        assert fv.amount == 5000.0


class TestGeoDistance:
    def test_haversine_same_point_is_zero(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        dist = FraudFeatureExtractor._haversine_distance(0.0, 0.0, 0.0, 0.0)
        assert dist == 0.0

    def test_haversine_known_distance(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        # Nairobi to Mombasa (~440 km)
        dist = FraudFeatureExtractor._haversine_distance(-1.286389, 36.817223, -4.043477, 39.668206)
        assert 430 < dist < 460

    def test_haversine_antipodal(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        # Opposite sides of the earth (~20000 km)
        dist = FraudFeatureExtractor._haversine_distance(0.0, 0.0, 0.0, 180.0)
        assert 20000 < dist < 20100

    @pytest.mark.asyncio
    async def test_geo_distance_with_last_known_location(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        # Pre-store last known location in Redis
        last_geo = json.dumps({"lat": -1.286389, "lng": 36.817223})

        redis_client = AsyncMock()

        async def mock_get(key):
            if "last_geo" in key:
                return last_geo
            return None

        redis_client.get = AsyncMock(side_effect=mock_get)
        redis_client.set = AsyncMock()
        extractor = FraudFeatureExtractor(redis_client=redis_client)

        # Request from Mombasa
        geo = GeoLocation(country="KE", lat=-4.043477, lng=39.668206)
        req = _make_fraud_request(geo_location=geo)
        fv = await extractor.extract(req)

        # Should compute distance Nairobi -> Mombasa (~440 km)
        assert fv.geo_distance_from_last_tx > 400

    @pytest.mark.asyncio
    async def test_geo_distance_zero_without_last_location(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        extractor = FraudFeatureExtractor(redis_client=redis_client)

        geo = GeoLocation(country="KE", lat=-1.286389, lng=36.817223)
        req = _make_fraud_request(geo_location=geo)
        fv = await extractor.extract(req)
        assert fv.geo_distance_from_last_tx == 0.0


class TestAMLFeatureExtractor:
    @pytest.mark.asyncio
    async def test_extract_returns_22_features(self):
        from feature_engineering.aml_features import AMLFeatureExtractor

        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.sismember = AsyncMock(return_value=True)

        pipe_mock = AsyncMock()
        pipe_mock.get = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[None, None, None])
        redis_client.pipeline.return_value = pipe_mock

        extractor = AMLFeatureExtractor(redis_client=redis_client, db_pool=None)
        req = _make_aml_request()
        fv = await extractor.extract(req)
        arr = fv.to_array()
        assert len(arr) == 22


class TestIPCountryMismatch:
    @pytest.mark.asyncio
    async def test_ip_country_mismatch_detected(self):
        from feature_engineering.aml_features import AMLFeatureExtractor

        redis_client = AsyncMock()

        async def mock_get(key):
            if key.startswith("ip_geo:"):
                return b"NG"  # Nigeria
            return None

        redis_client.get = AsyncMock(side_effect=mock_get)
        redis_client.sismember = AsyncMock(return_value=True)

        pipe_mock = AsyncMock()
        pipe_mock.get = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[None, None, None])
        redis_client.pipeline.return_value = pipe_mock

        extractor = AMLFeatureExtractor(redis_client=redis_client, db_pool=None)

        geo = GeoLocation(country="KE", lat=-1.286389, lng=36.817223)
        req = _make_aml_request(geo_location=geo, ip_address="1.2.3.4")
        fv = await extractor.extract(req)

        assert fv.ip_country_mismatch == 1

    @pytest.mark.asyncio
    async def test_ip_country_match_returns_zero(self):
        from feature_engineering.aml_features import AMLFeatureExtractor

        redis_client = AsyncMock()

        async def mock_get(key):
            if key.startswith("ip_geo:"):
                return b"KE"
            return None

        redis_client.get = AsyncMock(side_effect=mock_get)
        redis_client.sismember = AsyncMock(return_value=True)

        pipe_mock = AsyncMock()
        pipe_mock.get = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[None, None, None])
        redis_client.pipeline.return_value = pipe_mock

        extractor = AMLFeatureExtractor(redis_client=redis_client, db_pool=None)

        geo = GeoLocation(country="KE", lat=-1.286389, lng=36.817223)
        req = _make_aml_request(geo_location=geo, ip_address="1.2.3.4")
        fv = await extractor.extract(req)

        assert fv.ip_country_mismatch == 0

    @pytest.mark.asyncio
    async def test_ip_country_fallback_no_data(self):
        from feature_engineering.aml_features import AMLFeatureExtractor

        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.sismember = AsyncMock(return_value=True)

        pipe_mock = AsyncMock()
        pipe_mock.get = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[None, None, None])
        redis_client.pipeline.return_value = pipe_mock

        extractor = AMLFeatureExtractor(redis_client=redis_client, db_pool=None)

        geo = GeoLocation(country="KE", lat=-1.286389, lng=36.817223)
        req = _make_aml_request(geo_location=geo, ip_address="1.2.3.4")
        fv = await extractor.extract(req)

        # No IP geo data -> safe fallback to 0
        assert fv.ip_country_mismatch == 0


class TestVelocityCounterUpdates:
    @pytest.mark.asyncio
    async def test_fraud_velocity_counters(self):
        from feature_engineering.fraud_features import FraudFeatureExtractor

        pipe_mock = MagicMock()
        pipe_mock.incr = MagicMock(return_value=pipe_mock)
        pipe_mock.expire = MagicMock(return_value=pipe_mock)
        pipe_mock.incrbyfloat = MagicMock(return_value=pipe_mock)
        pipe_mock.execute = AsyncMock(return_value=[1, True, 1, True, 5000.0, True])

        redis_client = MagicMock()
        redis_client.pipeline = MagicMock(return_value=pipe_mock)

        extractor = FraudFeatureExtractor(redis_client=redis_client)
        await extractor.update_velocity_counters("user123", 5000.0)

        pipe_mock.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_aml_velocity_counters(self):
        from feature_engineering.aml_features import AMLFeatureExtractor

        pipe_mock = MagicMock()
        pipe_mock.incr = MagicMock(return_value=pipe_mock)
        pipe_mock.expire = MagicMock(return_value=pipe_mock)
        pipe_mock.incrbyfloat = MagicMock(return_value=pipe_mock)
        pipe_mock.execute = AsyncMock(return_value=[1, True, 1, True, 50000.0, True])

        redis_client = MagicMock()
        redis_client.pipeline = MagicMock(return_value=pipe_mock)

        extractor = AMLFeatureExtractor(redis_client=redis_client)
        await extractor.update_velocity_counters("user123", 50000.0)

        pipe_mock.execute.assert_called_once()
