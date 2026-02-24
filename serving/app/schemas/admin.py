"""Admin Schemas — Threshold management, alerts, and audit logging."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ThresholdUpdateRequest(BaseModel):
    model_type: str = Field(..., pattern="^(fraud|aml|merchant)$")
    thresholds: dict[str, float] = Field(
        ...,
        description="Key-value pairs of threshold names and values (0.0-1.0)",
    )
    reason: str = Field(..., min_length=5, max_length=500)


class ThresholdUpdateResponse(BaseModel):
    model_type: str
    previous: dict[str, float]
    updated: dict[str, float]
    version: str
    updated_at: datetime
    audit_id: UUID


class ThresholdHistoryEntry(BaseModel):
    audit_id: UUID
    model_type: str
    config_key: str
    old_value: float | None
    new_value: float
    changed_by: UUID | None
    reason: str | None
    created_at: datetime


class ThresholdHistoryResponse(BaseModel):
    model_type: str
    history: list[ThresholdHistoryEntry]
    total: int


class AlertConfigRequest(BaseModel):
    alert_type: str = Field(..., pattern="^(FRAUD_CRITICAL|AML_BLOCK|MERCHANT_BLOCKED|DRIFT_DETECTED|MODEL_FAILURE)$")
    channels: list[str] = Field(..., min_length=1, description="SLACK, EMAIL, PAGERDUTY, KAFKA")
    enabled: bool = True
    severity_threshold: str = Field("CRITICAL", pattern="^(INFO|WARNING|CRITICAL)$")


class AlertLogEntry(BaseModel):
    alert_id: UUID
    alert_type: str
    severity: str
    channel: str
    subject: str | None
    entity_type: str | None
    entity_id: str | None
    delivered: bool
    created_at: datetime


class AlertLogResponse(BaseModel):
    alerts: list[AlertLogEntry]
    total: int
