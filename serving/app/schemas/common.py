from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    MONITOR = "MONITOR"
    BLOCK = "BLOCK"


class GeoLocation(BaseModel):
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code")
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class FeatureContribution(BaseModel):
    feature: str
    value: float | None = None
    impact: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    models_loaded: dict[str, bool]
    timestamp: datetime


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
    correlation_id: str | None = None


class AuditEntry(BaseModel):
    entity_type: str
    entity_id: UUID
    action: str
    performed_by: UUID | None = None
    metadata: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
