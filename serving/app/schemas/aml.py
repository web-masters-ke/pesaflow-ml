from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .common import Decision, FeatureContribution, GeoLocation, RiskLevel

# === Request Schemas ===


class AMLScoreRequest(BaseModel):
    transaction_id: UUID
    user_id: UUID
    sender_wallet_id: UUID | None = None
    receiver_wallet_id: UUID | None = None
    amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    device_id: str | None = Field(None, max_length=255)
    ip_address: str | None = Field(None, max_length=45)
    geo_location: GeoLocation | None = None
    channel: str | None = Field(None, max_length=20, description="app/web/API")
    metadata: dict[str, Any] | None = None


class AMLBatchScoreRequest(BaseModel):
    transactions: list[AMLScoreRequest] = Field(..., min_length=1, max_length=1000)
    model_version: str | None = None


class AMLCaseCreateRequest(BaseModel):
    entity_type: str = Field(..., pattern="^(USER|TRANSACTION|MERCHANT)$")
    entity_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    trigger_reason: str
    priority: str = Field(default="HIGH", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")


# === Response Schemas ===


class AMLScoreResponse(BaseModel):
    transaction_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision: Decision
    model_version: str
    threshold_version: str
    top_risk_factors: list[str] = Field(default_factory=list)
    rule_overrides: list[str] = Field(default_factory=list)
    top_features: list[FeatureContribution] | None = None
    latency_ms: int
    correlation_id: str | None = None


class AMLExplanationResponse(BaseModel):
    transaction_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision: Decision
    top_features: list[FeatureContribution]
    explanation_method: str = "SHAP"
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class AMLBatchScoreResponse(BaseModel):
    total: int
    processed: int
    failed: int
    results: list[AMLScoreResponse]
    model_version: str
    duration_ms: int


class AMLCaseResponse(BaseModel):
    case_id: UUID
    entity_type: str
    entity_id: UUID
    trigger_reason: str
    risk_score: float
    priority: str
    status: str = "OPEN"
    created_at: datetime


class AMLUserRiskProfile(BaseModel):
    user_id: UUID
    cumulative_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_category: RiskLevel
    risk_trend: str
    velocity_1h: int = 0
    velocity_24h: int = 0
    total_volume_24h: float = 0.0
    network_risk_score: float = 0.0
    sanctions_flag: bool = False
    last_transaction_at: datetime | None = None


class SanctionsScreenResult(BaseModel):
    matched: bool
    match_type: str | None = None  # EXACT, FUZZY, COUNTRY
    matched_entity: str | None = None
    confidence: float = 0.0
    source: str | None = None


# === Internal Schemas ===


class AMLCaseUpdateRequest(BaseModel):
    status: str = Field(
        ..., pattern="^(OPEN|INVESTIGATING|ESCALATED|CLOSED_FALSE_POSITIVE|CLOSED_CONFIRMED|CLOSED_INCONCLUSIVE)$"
    )
    assigned_to: UUID | None = None
    notes: str | None = Field(None, max_length=2000)
    resolution: str | None = Field(None, max_length=500)


class AMLCaseListRequest(BaseModel):
    status: str | None = Field(
        None, pattern="^(OPEN|INVESTIGATING|ESCALATED|CLOSED_FALSE_POSITIVE|CLOSED_CONFIRMED|CLOSED_INCONCLUSIVE)$"
    )
    priority: str | None = Field(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    assigned_to: UUID | None = None
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class AMLCaseDetailResponse(BaseModel):
    case_id: UUID
    entity_type: str
    entity_id: UUID
    trigger_reason: str
    risk_score: float
    priority: str
    status: str
    assigned_to: UUID | None = None
    notes: str | None = None
    resolution: str | None = None
    created_at: datetime
    updated_at: datetime | None = None


class AMLCaseListResponse(BaseModel):
    total: int
    cases: list[AMLCaseDetailResponse]
    limit: int
    offset: int


class AMLFeatureVector(BaseModel):
    # Transaction level
    amount: float = 0.0
    velocity_1h: int = 0
    velocity_24h: int = 0
    total_volume_24h: float = 0.0
    avg_amount_30d: float = 0.0
    std_amount_30d: float = 0.0
    time_of_day: float = 0.0
    is_cross_border: int = 0
    # User level
    account_age_days: int = 0
    device_count_30d: int = 0
    ip_count_30d: int = 0
    new_device_flag: int = 0
    kyc_completeness_score: float = 1.0
    # Network level
    network_risk_score: float = 0.0
    circular_transfer_flag: int = 0
    shared_device_cluster_size: int = 0
    # Geographic
    high_risk_country_flag: int = 0
    sanctions_proximity_score: float = 0.0
    ip_country_mismatch: int = 0
    # Behavioral
    historical_structuring_flag: int = 0
    structuring_score_24h: float = 0.0
    rapid_drain_flag: int = 0

    def to_array(self) -> list[float]:
        return [
            self.amount,
            self.velocity_1h,
            self.velocity_24h,
            self.total_volume_24h,
            self.avg_amount_30d,
            self.std_amount_30d,
            self.time_of_day,
            self.is_cross_border,
            self.account_age_days,
            self.device_count_30d,
            self.ip_count_30d,
            self.new_device_flag,
            self.kyc_completeness_score,
            self.network_risk_score,
            self.circular_transfer_flag,
            self.shared_device_cluster_size,
            self.high_risk_country_flag,
            self.sanctions_proximity_score,
            self.ip_country_mismatch,
            self.historical_structuring_flag,
            self.structuring_score_24h,
            self.rapid_drain_flag,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
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
        ]
