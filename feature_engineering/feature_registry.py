"""Feature Registry — Versioned schema definitions for fraud, AML, and merchant features."""

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class FeatureSchema:
    name: str
    version: str
    feature_names: list[str]
    description: str = ""
    maturity_min_samples: int = 100  # Minimum samples before features are considered reliable
    maturity_min_coverage: float = 0.70  # Minimum % of features with non-default values

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def schema_hash(self) -> str:
        payload = json.dumps({"name": self.name, "version": self.version, "features": self.feature_names})
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def validate_features(self, feature_values: list[float]) -> bool:
        return len(feature_values) == self.feature_count

    def get_maturity_metadata(self) -> dict:
        """Return maturity-related metadata for this schema."""
        return {
            "schema_name": self.name,
            "schema_version": self.version,
            "feature_count": self.feature_count,
            "maturity_min_samples": self.maturity_min_samples,
            "maturity_min_coverage": self.maturity_min_coverage,
            "schema_hash": self.schema_hash,
        }


FRAUD_FEATURE_SCHEMA = FeatureSchema(
    name="fraud_features",
    version="1.0.0",
    description="Fraud detection feature vector — 13 signals across user, device, and transaction",
    feature_names=[
        "avg_transaction_amount_7d",
        "transaction_velocity_1h",
        "transaction_velocity_24h",
        "failed_login_attempts_24h",
        "account_age_days",
        "historical_fraud_flag",
        "device_risk_score",
        "device_fraud_count",
        "distinct_user_count",
        "amount",
        "geo_distance_from_last_tx",
        "time_of_day",
        "currency_risk",
    ],
)

AML_FEATURE_SCHEMA = FeatureSchema(
    name="aml_features",
    version="1.0.0",
    description="AML risk scoring feature vector — 22 signals across transaction, user, network, geographic, behavioral",
    feature_names=[
        "amount",
        "velocity_1h",
        "velocity_24h",
        "total_volume_24h",
        "avg_amount_30d",
        "std_amount_30d",
        "time_of_day",
        "is_cross_border",
        "account_age_days",
        "device_count_30d",
        "ip_count_30d",
        "new_device_flag",
        "kyc_completeness_score",
        "network_risk_score",
        "circular_transfer_flag",
        "shared_device_cluster_size",
        "high_risk_country_flag",
        "sanctions_proximity_score",
        "ip_country_mismatch",
        "historical_structuring_flag",
        "structuring_score_24h",
        "rapid_drain_flag",
    ],
)

MERCHANT_FEATURE_SCHEMA = FeatureSchema(
    name="merchant_features",
    version="1.0.0",
    description="Merchant risk feature vector — 15 signals across velocity, aggregate, and risk signals",
    feature_names=[
        "transaction_count_1h",
        "transaction_count_24h",
        "transaction_volume_24h",
        "unique_customers_24h",
        "avg_transaction_amount_30d",
        "std_transaction_amount_30d",
        "chargeback_rate_90d",
        "refund_rate_90d",
        "account_age_days",
        "fraud_transaction_rate",
        "high_risk_customer_ratio",
        "cross_border_ratio",
        "velocity_spike_flag",
        "mcc_risk_score",
        "avg_customer_risk_score",
    ],
)


@dataclass
class FeatureRegistryManager:
    schemas: dict[str, FeatureSchema] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.register("fraud", FRAUD_FEATURE_SCHEMA)
        self.register("aml", AML_FEATURE_SCHEMA)
        self.register("merchant", MERCHANT_FEATURE_SCHEMA)

    def register(self, key: str, schema: FeatureSchema) -> None:
        self.schemas[key] = schema

    def get(self, key: str) -> FeatureSchema | None:
        return self.schemas.get(key)

    def validate(self, key: str, feature_values: list[float]) -> bool:
        schema = self.get(key)
        if schema is None:
            return False
        return schema.validate_features(feature_values)


feature_registry = FeatureRegistryManager()
