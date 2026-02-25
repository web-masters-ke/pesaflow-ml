"""Unit tests for Data Maturity Detector."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from monitoring.data_maturity import DataMaturityDetector, MaturityLevel, MaturityReport


class TestMaturityLevelComputation:
    def test_cold_low_samples(self):
        detector = DataMaturityDetector()
        level = detector._compute_level(sample_count=50, feature_coverage=0.80, distribution_stability=0.5)
        assert level == MaturityLevel.COLD

    def test_cold_low_coverage(self):
        detector = DataMaturityDetector()
        level = detector._compute_level(sample_count=500, feature_coverage=0.60, distribution_stability=0.5)
        assert level == MaturityLevel.COLD

    def test_warming(self):
        detector = DataMaturityDetector()
        level = detector._compute_level(sample_count=500, feature_coverage=0.80, distribution_stability=0.5)
        assert level == MaturityLevel.WARMING

    def test_warm(self):
        detector = DataMaturityDetector()
        level = detector._compute_level(sample_count=5000, feature_coverage=0.95, distribution_stability=0.5)
        assert level == MaturityLevel.WARM

    def test_hot_stable_distributions(self):
        detector = DataMaturityDetector()
        level = detector._compute_level(sample_count=15000, feature_coverage=0.95, distribution_stability=0.05)
        assert level == MaturityLevel.HOT

    def test_warm_unstable_distributions(self):
        """Even with >10k samples, unstable distributions → WARM."""
        detector = DataMaturityDetector()
        level = detector._compute_level(sample_count=15000, feature_coverage=0.95, distribution_stability=0.3)
        assert level == MaturityLevel.WARM


class TestMaturityCaching:
    @pytest.mark.asyncio
    async def test_cached_level_from_redis(self):
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=b"HOT")

        detector = DataMaturityDetector(redis_client=redis_client)
        level = await detector.get_cached_level("fraud")
        assert level == MaturityLevel.HOT

    @pytest.mark.asyncio
    async def test_cached_level_fallback_to_memory(self):
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)

        detector = DataMaturityDetector(redis_client=redis_client)
        # No cache at all → COLD default
        level = await detector.get_cached_level("fraud")
        assert level == MaturityLevel.COLD

    @pytest.mark.asyncio
    async def test_cached_level_redis_failure_fallback(self):
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(side_effect=ConnectionError("Redis down"))

        detector = DataMaturityDetector(redis_client=redis_client)
        level = await detector.get_cached_level("fraud")
        assert level == MaturityLevel.COLD

    @pytest.mark.asyncio
    async def test_in_memory_cache_used(self):
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)

        detector = DataMaturityDetector(redis_client=redis_client)
        # Pre-populate in-memory cache
        detector._cache["fraud"] = MaturityReport(
            domain="fraud",
            level=MaturityLevel.WARM,
            sample_count=5000,
            positive_rate=0.05,
            feature_coverage=0.92,
            feature_staleness_hours=2.0,
            distribution_stability=0.08,
            label_delay_hours=24.0,
        )

        level = await detector.get_cached_level("fraud")
        assert level == MaturityLevel.WARM


class TestSignalGatheringFallbacks:
    @pytest.mark.asyncio
    async def test_sample_count_from_redis_fallback(self):
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=b"500")

        detector = DataMaturityDetector(redis_client=redis_client, db_pool=None)
        count = await detector._get_sample_count("fraud")
        assert count == 500

    @pytest.mark.asyncio
    async def test_sample_count_no_redis_returns_zero(self):
        detector = DataMaturityDetector(redis_client=None, db_pool=None)
        count = await detector._get_sample_count("fraud")
        assert count == 0

    @pytest.mark.asyncio
    async def test_positive_rate_no_db_fallback(self):
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=b"0.03")

        detector = DataMaturityDetector(redis_client=redis_client, db_pool=None)
        rate = await detector._get_positive_rate("fraud")
        assert rate == 0.03

    @pytest.mark.asyncio
    async def test_feature_coverage_no_data(self):
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)

        detector = DataMaturityDetector(redis_client=redis_client)
        coverage = await detector._get_feature_coverage("fraud")
        assert coverage == 0.0


class TestAlphaAndConfidence:
    def test_alpha_for_maturity(self):
        detector = DataMaturityDetector()
        assert detector.get_alpha_for_maturity(MaturityLevel.COLD) == 0.0
        assert detector.get_alpha_for_maturity(MaturityLevel.WARMING) == 0.3
        assert detector.get_alpha_for_maturity(MaturityLevel.WARM) == 0.7
        assert detector.get_alpha_for_maturity(MaturityLevel.HOT) == 1.0

    def test_confidence_floor_for_maturity(self):
        detector = DataMaturityDetector()
        assert detector.get_confidence_floor(MaturityLevel.COLD) == 0.1
        assert detector.get_confidence_floor(MaturityLevel.WARMING) == 0.3
        assert detector.get_confidence_floor(MaturityLevel.WARM) == 0.6
        assert detector.get_confidence_floor(MaturityLevel.HOT) == 0.8


class TestBuildDetails:
    def test_low_positive_rate_warning(self):
        detector = DataMaturityDetector()
        details = detector._build_details(
            "fraud", sample_count=1000, positive_rate=0.001, feature_coverage=0.9, label_delay=24
        )
        assert any("below expected" in w for w in details["warnings"])

    def test_high_positive_rate_warning(self):
        detector = DataMaturityDetector()
        details = detector._build_details(
            "fraud", sample_count=1000, positive_rate=0.5, feature_coverage=0.9, label_delay=24
        )
        assert any("above expected" in w for w in details["warnings"])

    def test_label_delay_warning(self):
        detector = DataMaturityDetector()
        details = detector._build_details(
            "fraud", sample_count=1000, positive_rate=0.05, feature_coverage=0.9, label_delay=100
        )
        assert any("Label delay" in w for w in details["warnings"])

    def test_low_coverage_recommendation(self):
        detector = DataMaturityDetector()
        details = detector._build_details(
            "fraud", sample_count=1000, positive_rate=0.05, feature_coverage=0.5, label_delay=24
        )
        assert any("coverage" in r.lower() for r in details["recommendations"])
