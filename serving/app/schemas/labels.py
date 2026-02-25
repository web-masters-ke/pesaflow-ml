"""Pydantic schemas for label feedback, active learning, and SSL."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field

# === Enums ===


class LabelSource(str, Enum):
    MANUAL_REVIEW = "MANUAL_REVIEW"
    CHARGEBACK_FEED = "CHARGEBACK_FEED"
    SAR_CONFIRMED = "SAR_CONFIRMED"
    AUTOMATED_RULE = "AUTOMATED_RULE"
    PARTNER_FEEDBACK = "PARTNER_FEEDBACK"
    SELF_TRAINING = "SELF_TRAINING"
    CO_TRAINING = "CO_TRAINING"
    TRI_TRAINING = "TRI_TRAINING"
    LABEL_PROPAGATION = "LABEL_PROPAGATION"


class Domain(str, Enum):
    FRAUD = "fraud"
    AML = "aml"
    MERCHANT = "merchant"


class ALStrategy(str, Enum):
    UNCERTAINTY = "UNCERTAINTY"
    MARGIN = "MARGIN"
    ENTROPY = "ENTROPY"
    DISAGREEMENT = "DISAGREEMENT"
    DENSITY_WEIGHTED = "DENSITY_WEIGHTED"


class ALQueueStatus(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    LABELED = "LABELED"
    EXPIRED = "EXPIRED"
    SKIPPED = "SKIPPED"


# === Label Submission ===


class LabelSubmitRequest(BaseModel):
    prediction_id: UUID
    domain: Domain
    label: int = Field(..., ge=0, le=1, description="0=legitimate, 1=fraudulent/suspicious")
    label_source: LabelSource
    labeled_by: UUID | None = None
    reason: str | None = Field(None, max_length=500)


class LabelBatchSubmitRequest(BaseModel):
    labels: list[LabelSubmitRequest] = Field(..., min_length=1, max_length=500)


class LabelSubmitResponse(BaseModel):
    prediction_id: UUID
    domain: Domain
    label: int
    label_source: LabelSource
    labeled_at: datetime
    previous_label: int | None = None


class LabelBatchSubmitResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[LabelSubmitResponse]
    errors: list[dict] = Field(default_factory=list)


# === Label Statistics ===


class DomainLabelStats(BaseModel):
    domain: Domain
    total_predictions: int = 0
    labeled_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    unlabeled_count: int = 0
    label_rate: float = 0.0
    maturity_level: str = "COLD"


class LabelStatisticsResponse(BaseModel):
    domains: list[DomainLabelStats]
    total_labeled: int = 0
    total_unlabeled: int = 0
    last_refreshed: datetime | None = None


# === Label Propagation ===


class LabelPropagationResponse(BaseModel):
    fraud_labels_propagated: int = 0
    aml_labels_propagated: int = 0
    total_propagated: int = 0


# === Active Learning Queue ===


class ALQueueItem(BaseModel):
    id: UUID
    domain: Domain
    prediction_id: UUID
    entity_id: UUID | None = None
    risk_score: float | None = None
    informativeness_score: float
    strategy: ALStrategy
    status: ALQueueStatus = ALQueueStatus.PENDING
    assigned_to: UUID | None = None
    expires_at: datetime | None = None
    created_at: datetime


class ALQueueResponse(BaseModel):
    domain: Domain
    total: int
    items: list[ALQueueItem]
    strategy: ALStrategy
    budget_remaining_today: int = 0


class ALRefreshResponse(BaseModel):
    domains_refreshed: list[str]
    items_added: dict[str, int] = Field(default_factory=dict)
    items_expired: dict[str, int] = Field(default_factory=dict)


# === Active Learning Config ===


class ALConfigResponse(BaseModel):
    domain: Domain
    strategy: ALStrategy
    daily_budget: int
    weekly_budget: int
    uncertainty_threshold: float
    is_active: bool
    updated_at: datetime | None = None


class ALConfigUpdateRequest(BaseModel):
    strategy: ALStrategy | None = None
    daily_budget: int | None = Field(None, ge=1, le=10000)
    weekly_budget: int | None = Field(None, ge=1, le=50000)
    uncertainty_threshold: float | None = Field(None, ge=0.0, le=1.0)
    is_active: bool | None = None


# === Active Learning Metrics ===


class ALMetricsResponse(BaseModel):
    domain: Domain
    total_queued: int = 0
    total_labeled_from_queue: int = 0
    labels_today: int = 0
    labels_this_week: int = 0
    daily_budget: int = 0
    weekly_budget: int = 0
    avg_informativeness: float = 0.0
    label_yield_rate: float = 0.0
    strategy: ALStrategy = ALStrategy.UNCERTAINTY


# === Create Cases ===


class ALCreateCasesRequest(BaseModel):
    domain: Domain
    count: int = Field(10, ge=1, le=100)


class ALCreateCasesResponse(BaseModel):
    domain: Domain
    cases_created: int = 0
    prediction_ids: list[UUID] = Field(default_factory=list)
