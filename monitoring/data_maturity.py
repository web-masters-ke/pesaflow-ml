"""Data Maturity Detector — Assesses data readiness per model domain to drive system behavior."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger


class MaturityLevel(str, Enum):
    """Data maturity levels that drive system behavior."""

    COLD = "COLD"        # <100 labeled samples, <70% feature coverage → pure rules
    WARMING = "WARMING"  # 100-1000 samples, 70-90% coverage → hybrid
    WARM = "WARM"        # 1000-10000 samples, >90% coverage → full ML
    HOT = "HOT"          # >10000 samples, stable distributions → ensemble


@dataclass
class MaturityReport:
    """Output of a maturity assessment."""

    domain: str
    level: MaturityLevel
    sample_count: int
    positive_rate: float
    feature_coverage: float
    feature_staleness_hours: float
    distribution_stability: float  # PSI vs baseline
    label_delay_hours: float
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "level": self.level.value,
            "sample_count": self.sample_count,
            "positive_rate": round(self.positive_rate, 4),
            "feature_coverage": round(self.feature_coverage, 4),
            "feature_staleness_hours": round(self.feature_staleness_hours, 2),
            "distribution_stability": round(self.distribution_stability, 4),
            "label_delay_hours": round(self.label_delay_hours, 2),
            "timestamp": self.timestamp,
            "details": self.details,
        }


# Thresholds for maturity level transitions
MATURITY_THRESHOLDS = {
    "cold_max_samples": 100,
    "warming_max_samples": 1000,
    "warm_max_samples": 10000,
    "cold_max_coverage": 0.70,
    "warming_max_coverage": 0.90,
    "psi_stable_threshold": 0.1,
    "positive_rate_min": 0.005,  # Flag if below 0.5%
    "positive_rate_max": 0.50,   # Flag if above 50%
}

# Expected positive rates per domain
DOMAIN_EXPECTED_RATES = {
    "fraud": (0.01, 0.10),     # 1-10% fraud
    "aml": (0.005, 0.08),      # 0.5-8% suspicious
    "merchant": (0.01, 0.10),  # 1-10% risky
}


class DataMaturityDetector:
    """Assesses data volume, feature coverage, label quality, and distribution stability per domain."""

    def __init__(
        self,
        redis_client: Any = None,
        db_pool: Any = None,
        thresholds: dict | None = None,
        check_interval_seconds: int = 3600,
    ):
        self._redis = redis_client
        self._db = db_pool
        self._thresholds = {**MATURITY_THRESHOLDS, **(thresholds or {})}
        self._check_interval = check_interval_seconds
        self._cache: dict[str, MaturityReport] = {}

    async def assess(self, domain: str) -> MaturityReport:
        """Run full maturity assessment for a domain.

        Args:
            domain: One of 'fraud', 'aml', 'merchant'

        Returns:
            MaturityReport with computed level and all signals
        """
        # Check cache
        cached = self._cache.get(domain)
        if cached and (time.time() - cached.timestamp) < self._check_interval:
            return cached

        logger.info(f"Assessing data maturity for domain: {domain}")

        # Gather signals
        sample_count = await self._get_sample_count(domain)
        positive_rate = await self._get_positive_rate(domain)
        feature_coverage = await self._get_feature_coverage(domain)
        feature_staleness = await self._get_feature_staleness(domain)
        distribution_stability = await self._get_distribution_stability(domain)
        label_delay = await self._get_label_delay(domain)

        # Compute maturity level
        level = self._compute_level(
            sample_count=sample_count,
            feature_coverage=feature_coverage,
            distribution_stability=distribution_stability,
        )

        # Build details with warnings
        details = self._build_details(
            domain=domain,
            sample_count=sample_count,
            positive_rate=positive_rate,
            feature_coverage=feature_coverage,
            label_delay=label_delay,
        )

        report = MaturityReport(
            domain=domain,
            level=level,
            sample_count=sample_count,
            positive_rate=positive_rate,
            feature_coverage=feature_coverage,
            feature_staleness_hours=feature_staleness,
            distribution_stability=distribution_stability,
            label_delay_hours=label_delay,
            details=details,
        )

        # Cache locally
        self._cache[domain] = report

        # Store in Redis for real-time access
        await self._store_maturity(report)

        logger.info(f"Maturity assessment for {domain}: {level.value} "
                     f"(samples={sample_count}, coverage={feature_coverage:.2%})")

        return report

    def _compute_level(
        self,
        sample_count: int,
        feature_coverage: float,
        distribution_stability: float,
    ) -> MaturityLevel:
        """Determine maturity level from signals."""
        t = self._thresholds

        if sample_count < t["cold_max_samples"] or feature_coverage < t["cold_max_coverage"]:
            return MaturityLevel.COLD

        if sample_count < t["warming_max_samples"] or feature_coverage < t["warming_max_coverage"]:
            return MaturityLevel.WARMING

        if sample_count < t["warm_max_samples"]:
            return MaturityLevel.WARM

        # HOT requires stable distributions
        if distribution_stability <= t["psi_stable_threshold"]:
            return MaturityLevel.HOT

        # >10k samples but unstable distributions → WARM
        return MaturityLevel.WARM

    def _build_details(
        self,
        domain: str,
        sample_count: int,
        positive_rate: float,
        feature_coverage: float,
        label_delay: float,
    ) -> dict:
        """Build details dict with warnings and recommendations."""
        details: dict[str, Any] = {"warnings": [], "recommendations": []}

        # Positive rate checks
        expected_min, expected_max = DOMAIN_EXPECTED_RATES.get(domain, (0.01, 0.10))
        if positive_rate < expected_min and sample_count > 0:
            details["warnings"].append(
                f"Positive rate {positive_rate:.2%} below expected minimum {expected_min:.2%}"
            )
            details["recommendations"].append("Verify labeling pipeline is working correctly")
        elif positive_rate > expected_max:
            details["warnings"].append(
                f"Positive rate {positive_rate:.2%} above expected maximum {expected_max:.2%}"
            )
            details["recommendations"].append("Check for labeling bias or data quality issues")

        # Feature coverage
        if feature_coverage < 0.70:
            details["recommendations"].append("Improve feature extraction pipeline coverage")

        # Label delay
        if label_delay > 72:
            details["warnings"].append(f"Label delay {label_delay:.1f}h exceeds 72h target")
            details["recommendations"].append("Speed up feedback loop for label confirmation")

        return details

    async def _get_sample_count(self, domain: str) -> int:
        """Get total labeled samples for a domain."""
        if not self._db:
            return await self._get_sample_count_from_redis(domain)

        table_map = {
            "fraud": "ml_predictions",
            "aml": "aml_predictions",
            "merchant": "merchant_risk_predictions",
        }
        table = table_map.get(domain)
        if not table:
            return 0

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(f"SELECT COUNT(*) as cnt FROM {table} WHERE label IS NOT NULL")
                return int(row["cnt"]) if row else 0
        except Exception as e:
            logger.warning(f"Failed to get sample count for {domain}: {e}")
            return await self._get_sample_count_from_redis(domain)

    async def _get_sample_count_from_redis(self, domain: str) -> int:
        """Fallback: get sample count from Redis counter."""
        if not self._redis:
            return 0
        try:
            count = await self._redis.get(f"pesaflow:maturity:{domain}:sample_count")
            return int(count) if count else 0
        except Exception:
            return 0

    async def _get_positive_rate(self, domain: str) -> float:
        """Get percentage of positive labels."""
        if not self._db:
            return await self._get_positive_rate_from_redis(domain)

        table_map = {
            "fraud": ("ml_predictions", "label"),
            "aml": ("aml_predictions", "label"),
            "merchant": ("merchant_risk_predictions", "label"),
        }
        table, label_col = table_map.get(domain, ("", ""))
        if not table:
            return 0.0

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT AVG(CASE WHEN {label_col} = 1 THEN 1.0 ELSE 0.0 END) as rate "
                    f"FROM {table} WHERE {label_col} IS NOT NULL"
                )
                return float(row["rate"]) if row and row["rate"] is not None else 0.0
        except Exception as e:
            logger.warning(f"Failed to get positive rate for {domain}: {e}")
            return await self._get_positive_rate_from_redis(domain)

    async def _get_positive_rate_from_redis(self, domain: str) -> float:
        """Fallback: get positive rate from Redis."""
        if not self._redis:
            return 0.0
        try:
            rate = await self._redis.get(f"pesaflow:maturity:{domain}:positive_rate")
            return float(rate) if rate else 0.0
        except Exception:
            return 0.0

    async def _get_feature_coverage(self, domain: str) -> float:
        """Get percentage of features with non-default values in recent window."""
        if not self._redis:
            return 0.0

        try:
            coverage = await self._redis.get(f"pesaflow:maturity:{domain}:feature_coverage")
            if coverage:
                return float(coverage)

            # Compute from recent predictions if available
            coverage_data = await self._redis.get(f"pesaflow:features:{domain}:coverage_stats")
            if coverage_data:
                import json
                stats = json.loads(coverage_data)
                total_features = stats.get("total", 1)
                covered = stats.get("non_default", 0)
                return covered / total_features

            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get feature coverage for {domain}: {e}")
            return 0.0

    async def _get_feature_staleness(self, domain: str) -> float:
        """Get median age of cached feature data in hours."""
        if not self._redis:
            return 0.0

        try:
            staleness = await self._redis.get(f"pesaflow:maturity:{domain}:feature_staleness_hours")
            return float(staleness) if staleness else 0.0
        except Exception:
            return 0.0

    async def _get_distribution_stability(self, domain: str) -> float:
        """Get PSI of recent vs baseline feature distributions."""
        if not self._redis:
            return 1.0  # Assume unstable when no data

        try:
            psi = await self._redis.get(f"pesaflow:maturity:{domain}:distribution_psi")
            return float(psi) if psi else 1.0
        except Exception:
            return 1.0

    async def _get_label_delay(self, domain: str) -> float:
        """Get average time between transaction and label confirmation in hours."""
        if not self._db:
            return 0.0

        table_map = {
            "fraud": "ml_predictions",
            "aml": "aml_predictions",
            "merchant": "merchant_risk_predictions",
        }
        table = table_map.get(domain)
        if not table:
            return 0.0

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT AVG(EXTRACT(EPOCH FROM (labeled_at - created_at)) / 3600) as avg_delay "
                    f"FROM {table} WHERE labeled_at IS NOT NULL "
                    f"AND created_at > NOW() - INTERVAL '7 days'"
                )
                return float(row["avg_delay"]) if row and row["avg_delay"] is not None else 0.0
        except Exception as e:
            logger.warning(f"Failed to get label delay for {domain}: {e}")
            return 0.0

    async def _store_maturity(self, report: MaturityReport) -> None:
        """Store maturity report in Redis for real-time access."""
        if not self._redis:
            return

        try:
            import json
            key = f"pesaflow:maturity:{report.domain}"
            await self._redis.set(key, json.dumps(report.to_dict()), ex=self._check_interval * 2)
            await self._redis.set(f"{key}:level", report.level.value, ex=self._check_interval * 2)
        except Exception as e:
            logger.warning(f"Failed to store maturity in Redis: {e}")

    async def get_cached_level(self, domain: str) -> MaturityLevel:
        """Get maturity level from Redis cache (fast path for scoring)."""
        if not self._redis:
            # Fallback to in-memory cache
            cached = self._cache.get(domain)
            return cached.level if cached else MaturityLevel.COLD

        try:
            level_str = await self._redis.get(f"pesaflow:maturity:{domain}:level")
            if level_str:
                return MaturityLevel(level_str.decode() if isinstance(level_str, bytes) else level_str)
        except Exception:
            pass

        # Fallback to in-memory
        cached = self._cache.get(domain)
        return cached.level if cached else MaturityLevel.COLD

    async def get_all_maturity(self) -> dict[str, MaturityReport]:
        """Get maturity reports for all domains."""
        reports = {}
        for domain in ("fraud", "aml", "merchant"):
            reports[domain] = await self.assess(domain)
        return reports

    def get_alpha_for_maturity(self, level: MaturityLevel) -> float:
        """Get blending alpha (supervised weight) based on maturity level.

        Returns value from 0.0 (full anomaly/rules) to 1.0 (full supervised ML).
        """
        return {
            MaturityLevel.COLD: 0.0,
            MaturityLevel.WARMING: 0.3,
            MaturityLevel.WARM: 0.7,
            MaturityLevel.HOT: 1.0,
        }[level]

    def get_confidence_floor(self, level: MaturityLevel) -> float:
        """Get minimum confidence floor based on maturity level."""
        return {
            MaturityLevel.COLD: 0.1,
            MaturityLevel.WARMING: 0.3,
            MaturityLevel.WARM: 0.6,
            MaturityLevel.HOT: 0.8,
        }[level]
