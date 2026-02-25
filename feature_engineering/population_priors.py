"""Population Priors — Computed population-based feature defaults with Bayesian blending."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from feature_engineering.feature_registry import (
    AML_FEATURE_SCHEMA,
    FRAUD_FEATURE_SCHEMA,
    MERCHANT_FEATURE_SCHEMA,
    FeatureSchema,
)

# Hardcoded fallback defaults when zero data exists
HARDCODED_DEFAULTS: dict[str, dict[str, float]] = {
    "fraud": {
        "avg_transaction_amount_7d": 50.0,
        "transaction_velocity_1h": 1.0,
        "transaction_velocity_24h": 5.0,
        "failed_login_attempts_24h": 0.0,
        "account_age_days": 90.0,
        "historical_fraud_flag": 0.0,
        "device_risk_score": 0.1,
        "device_fraud_count": 0.0,
        "distinct_user_count": 1.0,
        "amount": 50.0,
        "geo_distance_from_last_tx": 0.0,
        "time_of_day": 12.0,
        "currency_risk": 0.0,
    },
    "aml": {
        "amount": 100.0,
        "velocity_1h": 1.0,
        "velocity_24h": 3.0,
        "total_volume_24h": 300.0,
        "avg_amount_30d": 100.0,
        "std_amount_30d": 50.0,
        "time_of_day": 12.0,
        "is_cross_border": 0.0,
        "account_age_days": 90.0,
        "device_count_30d": 1.0,
        "ip_count_30d": 2.0,
        "new_device_flag": 0.0,
        "kyc_completeness_score": 0.7,
        "network_risk_score": 0.1,
        "circular_transfer_flag": 0.0,
        "shared_device_cluster_size": 1.0,
        "high_risk_country_flag": 0.0,
        "sanctions_proximity_score": 0.0,
        "ip_country_mismatch": 0.0,
        "historical_structuring_flag": 0.0,
        "structuring_score_24h": 0.0,
        "rapid_drain_flag": 0.0,
    },
    "merchant": {
        "transaction_count_1h": 5.0,
        "transaction_count_24h": 50.0,
        "transaction_volume_24h": 5000.0,
        "unique_customers_24h": 20.0,
        "avg_transaction_amount_30d": 100.0,
        "std_transaction_amount_30d": 50.0,
        "chargeback_rate_90d": 0.01,
        "refund_rate_90d": 0.03,
        "account_age_days": 180.0,
        "fraud_transaction_rate": 0.005,
        "high_risk_customer_ratio": 0.05,
        "cross_border_ratio": 0.1,
        "velocity_spike_flag": 0.0,
        "mcc_risk_score": 0.1,
        "avg_customer_risk_score": 0.1,
    },
}

DOMAIN_SCHEMAS: dict[str, FeatureSchema] = {
    "fraud": FRAUD_FEATURE_SCHEMA,
    "aml": AML_FEATURE_SCHEMA,
    "merchant": MERCHANT_FEATURE_SCHEMA,
}


@dataclass
class PopulationStats:
    """Computed statistics for a single feature."""

    feature_name: str
    median: float
    mean: float
    mode: float
    std: float
    sample_count: int
    updated_at: float = field(default_factory=time.time)


class PopulationPriors:
    """Manages population-based feature priors with Bayesian updating."""

    def __init__(
        self,
        redis_client: Any = None,
        db_pool: Any = None,
        prior_ttl_seconds: int = 3600,
        recompute_interval_seconds: int = 1800,
    ):
        self._redis = redis_client
        self._db = db_pool
        self._prior_ttl = prior_ttl_seconds
        self._recompute_interval = recompute_interval_seconds

        # In-memory cache: domain -> feature_name -> PopulationStats
        self._stats_cache: dict[str, dict[str, PopulationStats]] = {}
        self._last_compute: dict[str, float] = {}

    async def initialize(self, domain: str) -> None:
        """Compute and cache population priors for a domain on startup."""
        logger.info(f"Initializing population priors for {domain}")
        await self._compute_priors(domain)

    async def get_prior(self, domain: str, feature_name: str) -> float:
        """Get population prior for a specific feature.

        Returns:
            Computed population median if available, else hardcoded default.
        """
        # Check if recompute needed
        last = self._last_compute.get(domain, 0)
        if time.time() - last > self._recompute_interval:
            await self._compute_priors(domain)

        # Try computed stats first
        domain_stats = self._stats_cache.get(domain, {})
        if feature_name in domain_stats:
            return domain_stats[feature_name].median

        # Try Redis
        value = await self._get_prior_from_redis(domain, feature_name)
        if value is not None:
            return value

        # Fall back to hardcoded defaults
        return HARDCODED_DEFAULTS.get(domain, {}).get(feature_name, 0.0)

    async def get_all_priors(self, domain: str) -> dict[str, float]:
        """Get all population priors for a domain as a dict."""
        schema = DOMAIN_SCHEMAS.get(domain)
        if not schema:
            return HARDCODED_DEFAULTS.get(domain, {})

        priors = {}
        for feature_name in schema.feature_names:
            priors[feature_name] = await self.get_prior(domain, feature_name)
        return priors

    def blend_with_user_data(
        self,
        domain: str,
        feature_name: str,
        population_prior: float,
        user_value: float,
        user_sample_count: int,
        prior_strength: int = 10,
    ) -> float:
        """Bayesian blending of population prior with user-specific values.

        Uses a simple weighted average where the weight shifts from population
        prior toward user data as user_sample_count increases.

        Args:
            domain: Model domain
            feature_name: Feature name
            population_prior: Population-level prior value
            user_value: User's observed value
            user_sample_count: Number of observations for this user
            prior_strength: Pseudo-count for the prior (higher = stickier prior)

        Returns:
            Blended value that transitions from prior → user-specific
        """
        if user_sample_count <= 0:
            return population_prior

        # Bayesian posterior mean with conjugate prior
        # weight = user_samples / (user_samples + prior_strength)
        weight = user_sample_count / (user_sample_count + prior_strength)
        blended = (1 - weight) * population_prior + weight * user_value

        return blended

    def get_feature_defaults(self, domain: str) -> list[float]:
        """Get ordered list of default feature values for a domain.

        Used when a feature vector has missing values.
        """
        schema = DOMAIN_SCHEMAS.get(domain)
        if not schema:
            return []

        domain_stats = self._stats_cache.get(domain, {})
        defaults = []
        hardcoded = HARDCODED_DEFAULTS.get(domain, {})

        for feature_name in schema.feature_names:
            if feature_name in domain_stats:
                defaults.append(domain_stats[feature_name].median)
            else:
                defaults.append(hardcoded.get(feature_name, 0.0))

        return defaults

    async def _compute_priors(self, domain: str) -> None:
        """Compute population statistics from available data."""
        if not self._db:
            # No database, try loading from Redis
            await self._load_priors_from_redis(domain)
            return

        schema = DOMAIN_SCHEMAS.get(domain)
        if not schema:
            return

        table_map = {
            "fraud": "ml_prediction_feature_snapshot",
            "aml": "aml_predictions",
            "merchant": "merchant_risk_predictions",
        }
        table = table_map.get(domain)
        if not table:
            return

        try:
            async with self._db.acquire() as conn:
                # Fetch recent feature snapshots
                if domain == "fraud":
                    rows = await conn.fetch(f"SELECT feature_data FROM {table} " "ORDER BY created_at DESC LIMIT 10000")
                    data_key = "feature_data"
                else:
                    rows = await conn.fetch(
                        f"SELECT feature_snapshot FROM {table} " "ORDER BY created_at DESC LIMIT 10000"
                    )
                    data_key = "feature_snapshot"

                if not rows:
                    logger.info(f"No data available for {domain} priors, using hardcoded defaults")
                    self._last_compute[domain] = time.time()
                    return

                # Parse feature vectors
                all_values: dict[str, list[float]] = {name: [] for name in schema.feature_names}
                for row in rows:
                    snapshot = row[data_key]
                    if isinstance(snapshot, dict):
                        for name in schema.feature_names:
                            if name in snapshot and snapshot[name] is not None:
                                all_values[name].append(float(snapshot[name]))
                    elif isinstance(snapshot, (list, tuple)):
                        for i, name in enumerate(schema.feature_names):
                            if i < len(snapshot) and snapshot[i] is not None:
                                all_values[name].append(float(snapshot[i]))

                # Compute statistics
                domain_stats: dict[str, PopulationStats] = {}
                for name, values in all_values.items():
                    if len(values) < 5:
                        continue

                    arr = np.array(values)
                    # Compute mode as the value closest to the median for continuous features
                    hist, bin_edges = np.histogram(arr, bins=min(50, len(values) // 2 + 1))
                    mode_bin = np.argmax(hist)
                    mode_val = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2

                    domain_stats[name] = PopulationStats(
                        feature_name=name,
                        median=float(np.median(arr)),
                        mean=float(np.mean(arr)),
                        mode=float(mode_val),
                        std=float(np.std(arr)),
                        sample_count=len(values),
                    )

                self._stats_cache[domain] = domain_stats
                self._last_compute[domain] = time.time()

                # Store in Redis
                await self._store_priors_to_redis(domain, domain_stats)

                logger.info(
                    f"Computed population priors for {domain}: "
                    f"{len(domain_stats)}/{len(schema.feature_names)} features from {len(rows)} samples"
                )

        except Exception as e:
            logger.warning(f"Failed to compute priors for {domain}: {e}")
            self._last_compute[domain] = time.time()

    async def _store_priors_to_redis(self, domain: str, stats: dict[str, PopulationStats]) -> None:
        """Store computed priors in Redis with TTL."""
        if not self._redis:
            return

        try:
            data = {}
            for name, s in stats.items():
                data[name] = {
                    "median": s.median,
                    "mean": s.mean,
                    "mode": s.mode,
                    "std": s.std,
                    "sample_count": s.sample_count,
                    "updated_at": s.updated_at,
                }

            key = f"pesaflow:priors:{domain}"
            await self._redis.set(key, json.dumps(data), ex=self._prior_ttl)
        except Exception as e:
            logger.warning(f"Failed to store priors in Redis for {domain}: {e}")

    async def _load_priors_from_redis(self, domain: str) -> None:
        """Load priors from Redis cache."""
        if not self._redis:
            return

        try:
            key = f"pesaflow:priors:{domain}"
            data_raw = await self._redis.get(key)
            if not data_raw:
                return

            data = json.loads(data_raw)
            domain_stats: dict[str, PopulationStats] = {}
            for name, values in data.items():
                domain_stats[name] = PopulationStats(
                    feature_name=name,
                    median=values["median"],
                    mean=values["mean"],
                    mode=values["mode"],
                    std=values["std"],
                    sample_count=values["sample_count"],
                    updated_at=values.get("updated_at", time.time()),
                )

            self._stats_cache[domain] = domain_stats
            self._last_compute[domain] = time.time()
            logger.info(f"Loaded population priors for {domain} from Redis: {len(domain_stats)} features")
        except Exception as e:
            logger.warning(f"Failed to load priors from Redis for {domain}: {e}")

    async def _get_prior_from_redis(self, domain: str, feature_name: str) -> float | None:
        """Get a single prior value from Redis."""
        if not self._redis:
            return None

        try:
            key = f"pesaflow:priors:{domain}"
            data_raw = await self._redis.get(key)
            if not data_raw:
                return None

            data = json.loads(data_raw)
            if feature_name in data:
                return float(data[feature_name]["median"])
        except Exception:
            pass

        return None

    async def update_coverage_stats(self, domain: str, feature_vector: list[float]) -> float:
        """Update feature coverage statistics based on a new feature vector.

        Returns:
            Current feature coverage ratio (0.0-1.0)
        """
        schema = DOMAIN_SCHEMAS.get(domain)
        if not schema:
            return 0.0

        defaults = HARDCODED_DEFAULTS.get(domain, {})
        non_default_count = 0

        for i, name in enumerate(schema.feature_names):
            if i < len(feature_vector):
                default_val = defaults.get(name, 0.0)
                if abs(feature_vector[i] - default_val) > 1e-6:
                    non_default_count += 1

        coverage = non_default_count / schema.feature_count if schema.feature_count > 0 else 0.0

        # Store in Redis
        if self._redis:
            try:
                stats = json.dumps({"total": schema.feature_count, "non_default": non_default_count})
                await self._redis.set(
                    f"pesaflow:features:{domain}:coverage_stats",
                    stats,
                    ex=self._prior_ttl,
                )
                await self._redis.set(
                    f"pesaflow:maturity:{domain}:feature_coverage",
                    str(coverage),
                    ex=self._prior_ttl,
                )
            except Exception:
                pass

        return coverage
