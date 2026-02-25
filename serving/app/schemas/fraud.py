from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .common import Decision, FeatureContribution, GeoLocation, RiskLevel

# === Request Schemas ===


class FraudScoreRequest(BaseModel):
    transaction_id: UUID
    user_id: UUID
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(..., min_length=3, max_length=3, description="ISO currency code")
    transaction_type: str = Field(..., min_length=1, max_length=50)
    device_fingerprint: str | None = Field(None, max_length=255)
    ip_address: str | None = Field(None, max_length=45)
    geo_location: GeoLocation | None = None
    merchant_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] | None = None


class FraudBatchScoreRequest(BaseModel):
    transactions: list[FraudScoreRequest] = Field(..., min_length=1, max_length=1000)
    model_version: str | None = None


# === Response Schemas ===


class FraudScoreResponse(BaseModel):
    transaction_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision: Decision
    model_version: str
    threshold_version: str
    latency_ms: int
    rule_overrides: list[str] = Field(default_factory=list)
    top_features: list[FeatureContribution] | None = None
    correlation_id: str | None = None


class FraudExplanationResponse(BaseModel):
    transaction_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision: Decision
    top_features: list[FeatureContribution]
    explanation_method: str = "SHAP"
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    feature_snapshot_hash: str | None = None


class FraudBatchScoreResponse(BaseModel):
    total: int
    processed: int
    failed: int
    results: list[FraudScoreResponse]
    model_version: str
    duration_ms: int


# === Internal Schemas ===


class FraudFeatureVector(BaseModel):
    # User-based
    avg_transaction_amount_7d: float = 0.0
    transaction_velocity_1h: int = 0
    transaction_velocity_24h: int = 0
    failed_login_attempts_24h: int = 0
    account_age_days: int = 0
    historical_fraud_flag: int = 0
    # Device-based
    device_risk_score: float = 0.0
    device_fraud_count: int = 0
    distinct_user_count: int = 0
    # Transaction-based
    amount: float = 0.0
    geo_distance_from_last_tx: float = 0.0
    time_of_day: float = 0.0
    currency_risk: float = 0.0

    def to_array(self) -> list[float]:
        return [
            self.avg_transaction_amount_7d,
            self.transaction_velocity_1h,
            self.transaction_velocity_24h,
            self.failed_login_attempts_24h,
            self.account_age_days,
            self.historical_fraud_flag,
            self.device_risk_score,
            self.device_fraud_count,
            self.distinct_user_count,
            self.amount,
            self.geo_distance_from_last_tx,
            self.time_of_day,
            self.currency_risk,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
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
        ]


class OverrideFlags(BaseModel):
    blacklisted_device: bool = False
    blacklisted_ip: bool = False
    blacklisted_user: bool = False
    sanctioned_country: bool = False
    velocity_anomaly: bool = False
    new_account_high_amount: bool = False
