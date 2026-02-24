"""Merchant Risk Schemas — Pydantic models for merchant risk scoring."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .common import Decision, FeatureContribution, GeoLocation, RiskLevel


# === Enums ===


class MerchantTier(str):
    STANDARD = "STANDARD"
    ENHANCED = "ENHANCED"
    RESTRICTED = "RESTRICTED"
    BLOCKED = "BLOCKED"


# === Request Schemas ===


class MerchantScoreRequest(BaseModel):
    merchant_id: UUID
    transaction_id: UUID | None = None
    amount: float = Field(0.0, ge=0, description="Transaction amount triggering rescore")
    currency: str = Field("KES", min_length=3, max_length=3)
    customer_id: UUID | None = None
    mcc_code: str | None = Field(None, max_length=10, description="Merchant Category Code")
    geo_location: GeoLocation | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] | None = None


class MerchantBatchScoreRequest(BaseModel):
    merchants: list[MerchantScoreRequest] = Field(..., min_length=1, max_length=500)
    model_version: str | None = None


# === Response Schemas ===


class MerchantScoreResponse(BaseModel):
    merchant_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision: Decision
    merchant_tier: str
    model_version: str
    threshold_version: str
    top_risk_factors: list[str] = Field(default_factory=list)
    rule_overrides: list[str] = Field(default_factory=list)
    top_features: list[FeatureContribution] | None = None
    latency_ms: int
    correlation_id: str | None = None


class MerchantExplanationResponse(BaseModel):
    merchant_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision: Decision
    top_features: list[FeatureContribution]
    explanation_method: str = "SHAP"
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class MerchantBatchScoreResponse(BaseModel):
    total: int
    processed: int
    failed: int
    results: list[MerchantScoreResponse]
    model_version: str
    duration_ms: int


# === Internal Schemas ===


class MerchantFeatureVector(BaseModel):
    # Velocity features
    transaction_count_1h: int = 0
    transaction_count_24h: int = 0
    transaction_volume_24h: float = 0.0
    unique_customers_24h: int = 0
    # Aggregate features
    avg_transaction_amount_30d: float = 0.0
    std_transaction_amount_30d: float = 0.0
    chargeback_rate_90d: float = 0.0
    refund_rate_90d: float = 0.0
    account_age_days: int = 0
    # Risk signals
    fraud_transaction_rate: float = 0.0
    high_risk_customer_ratio: float = 0.0
    cross_border_ratio: float = 0.0
    velocity_spike_flag: int = 0
    mcc_risk_score: float = 0.0
    avg_customer_risk_score: float = 0.0

    def to_array(self) -> list[float]:
        return [
            float(self.transaction_count_1h),
            float(self.transaction_count_24h),
            self.transaction_volume_24h,
            float(self.unique_customers_24h),
            self.avg_transaction_amount_30d,
            self.std_transaction_amount_30d,
            self.chargeback_rate_90d,
            self.refund_rate_90d,
            float(self.account_age_days),
            self.fraud_transaction_rate,
            self.high_risk_customer_ratio,
            self.cross_border_ratio,
            float(self.velocity_spike_flag),
            self.mcc_risk_score,
            self.avg_customer_risk_score,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
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
        ]
