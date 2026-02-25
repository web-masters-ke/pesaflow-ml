"""Admin Configuration Routes — Threshold management, model info, drift status, alerts."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status

from serving.app.api.dependencies import ServiceContainer, get_container
from serving.app.middleware.auth import require_role
from serving.app.schemas.admin import (
    AlertLogEntry,
    AlertLogResponse,
    ThresholdHistoryEntry,
    ThresholdHistoryResponse,
    ThresholdUpdateRequest,
    ThresholdUpdateResponse,
)
from serving.app.services.alert_service import AlertService

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/models", summary="List loaded models and their status")
async def list_models(
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "FRAUD_ANALYST", "AML_AUDITOR")),
) -> dict:
    models = {}

    if container.fraud_model:
        models["fraud"] = container.fraud_model.get_metadata()
    if container.aml_model:
        models["aml"] = container.aml_model.get_metadata()
    if container.merchant_model:
        models["merchant"] = container.merchant_model.get_metadata()

    return {"models": models}


@router.get("/thresholds", summary="Get current threshold configuration")
async def get_thresholds(
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "FRAUD_ANALYST")),
) -> dict:
    fraud_thresholds = {}
    aml_thresholds = {}
    merchant_thresholds = {}

    if container.fraud_scoring_service:
        ft = container.fraud_scoring_service._decision.thresholds
        fraud_thresholds = {
            "approve_below": ft.approve_below,
            "review_above": ft.review_above,
            "block_above": ft.block_above,
            "version": ft.version,
        }

    if container.aml_scoring_service:
        at = container.aml_scoring_service._decision.thresholds
        aml_thresholds = {
            "medium_above": at.medium_above,
            "high_above": at.high_above,
            "critical_above": at.critical_above,
            "version": at.version,
        }

    if container.merchant_risk_service:
        mt = container.merchant_risk_service._decision.thresholds
        merchant_thresholds = {
            "standard_below": mt.standard_below,
            "enhanced_above": mt.enhanced_above,
            "restricted_above": mt.restricted_above,
            "blocked_above": mt.blocked_above,
            "version": mt.version,
        }

    return {"fraud": fraud_thresholds, "aml": aml_thresholds, "merchant": merchant_thresholds}


@router.put("/thresholds", response_model=ThresholdUpdateResponse, summary="Update thresholds with audit")
async def update_thresholds(
    request: ThresholdUpdateRequest,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> ThresholdUpdateResponse:
    previous = {}
    updated = {}
    version = ""

    if request.model_type == "fraud" and container.fraud_scoring_service:
        ft = container.fraud_scoring_service._decision.thresholds
        previous = {"approve_below": ft.approve_below, "review_above": ft.review_above, "block_above": ft.block_above}

        if "approve_below" in request.thresholds:
            ft.approve_below = request.thresholds["approve_below"]
        if "review_above" in request.thresholds:
            ft.review_above = request.thresholds["review_above"]
        if "block_above" in request.thresholds:
            ft.block_above = request.thresholds["block_above"]

        if not ft.validate():
            # Rollback
            ft.approve_below = previous["approve_below"]
            ft.review_above = previous["review_above"]
            ft.block_above = previous["block_above"]
            raise HTTPException(status_code=400, detail="Invalid thresholds: must be approve < review < block")

        ft.version = f"tv{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        version = ft.version
        updated = {"approve_below": ft.approve_below, "review_above": ft.review_above, "block_above": ft.block_above}

    elif request.model_type == "aml" and container.aml_scoring_service:
        at = container.aml_scoring_service._decision.thresholds
        previous = {"medium_above": at.medium_above, "high_above": at.high_above, "critical_above": at.critical_above}

        if "medium_above" in request.thresholds:
            at.medium_above = request.thresholds["medium_above"]
        if "high_above" in request.thresholds:
            at.high_above = request.thresholds["high_above"]
        if "critical_above" in request.thresholds:
            at.critical_above = request.thresholds["critical_above"]

        if not at.validate():
            at.medium_above = previous["medium_above"]
            at.high_above = previous["high_above"]
            at.critical_above = previous["critical_above"]
            raise HTTPException(status_code=400, detail="Invalid AML thresholds: must be medium < high < critical")

        at.version = f"atv{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        version = at.version
        updated = {"medium_above": at.medium_above, "high_above": at.high_above, "critical_above": at.critical_above}

    elif request.model_type == "merchant" and container.merchant_risk_service:
        mrt = container.merchant_risk_service._decision.thresholds
        previous = {
            "standard_below": mrt.standard_below,
            "enhanced_above": mrt.enhanced_above,
            "restricted_above": mrt.restricted_above,
            "blocked_above": mrt.blocked_above,
        }

        if "standard_below" in request.thresholds:
            mrt.standard_below = request.thresholds["standard_below"]
        if "enhanced_above" in request.thresholds:
            mrt.enhanced_above = request.thresholds["enhanced_above"]
        if "restricted_above" in request.thresholds:
            mrt.restricted_above = request.thresholds["restricted_above"]
        if "blocked_above" in request.thresholds:
            mrt.blocked_above = request.thresholds["blocked_above"]

        if not mrt.validate():
            mrt.standard_below = previous["standard_below"]
            mrt.enhanced_above = previous["enhanced_above"]
            mrt.restricted_above = previous["restricted_above"]
            mrt.blocked_above = previous["blocked_above"]
            raise HTTPException(status_code=400, detail="Invalid merchant thresholds")

        mrt.version = f"mtv{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        version = mrt.version
        updated = {
            "standard_below": mrt.standard_below,
            "enhanced_above": mrt.enhanced_above,
            "restricted_above": mrt.restricted_above,
            "blocked_above": mrt.blocked_above,
        }

    else:
        raise HTTPException(
            status_code=404, detail=f"Model type '{request.model_type}' not found or service not loaded"
        )

    # Audit log
    audit_id = uuid.uuid4()
    if container.db_pool:
        try:
            async with container.db_pool.acquire() as conn:
                for key, new_val in request.thresholds.items():
                    old_val = previous.get(key)
                    await conn.execute(
                        """
                        INSERT INTO threshold_config_audit
                        (id, model_type, config_key, old_value, new_value, reason)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        str(uuid.uuid4()),
                        request.model_type,
                        key,
                        old_val,
                        new_val,
                        request.reason,
                    )
        except Exception:
            pass  # Non-critical

    return ThresholdUpdateResponse(
        model_type=request.model_type,
        previous=previous,
        updated=updated,
        version=version,
        updated_at=datetime.utcnow(),
        audit_id=audit_id,
    )


@router.get("/thresholds/history", response_model=ThresholdHistoryResponse, summary="Get threshold change history")
async def get_threshold_history(
    model_type: str = Query(..., pattern="^(fraud|aml|merchant)$"),
    limit: int = Query(50, ge=1, le=200),
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> ThresholdHistoryResponse:
    if not container.db_pool:
        return ThresholdHistoryResponse(model_type=model_type, history=[], total=0)

    async with container.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, model_type, config_key, old_value, new_value, changed_by, reason, created_at
            FROM threshold_config_audit
            WHERE model_type = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            model_type,
            limit,
        )

        count_row = await conn.fetchrow(
            "SELECT COUNT(*) as total FROM threshold_config_audit WHERE model_type = $1",
            model_type,
        )

    history = [
        ThresholdHistoryEntry(
            audit_id=row["id"],
            model_type=row["model_type"],
            config_key=row["config_key"],
            old_value=float(row["old_value"]) if row["old_value"] is not None else None,
            new_value=float(row["new_value"]),
            changed_by=row.get("changed_by"),
            reason=row.get("reason"),
            created_at=row["created_at"],
        )
        for row in rows
    ]

    return ThresholdHistoryResponse(model_type=model_type, history=history, total=count_row["total"])


@router.get("/feature-importance/{model_name}", summary="Get feature importance for a model")
async def get_feature_importance(
    model_name: str,
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN", "FRAUD_ANALYST")),
) -> dict:
    model_map = {
        "fraud": container.fraud_model,
        "aml": container.aml_model,
        "merchant": container.merchant_model,
    }
    model = model_map.get(model_name)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_name}' not found")

    importance = model.get_feature_importance()
    return {
        "model": model_name,
        "scoring_mode": "ml" if model.is_loaded else "heuristic",
        "importance": importance,
    }


@router.get("/alerts", response_model=AlertLogResponse, summary="Get alert history")
async def get_alert_history(
    alert_type: str | None = None,
    severity: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    container: ServiceContainer = Depends(get_container),
    _auth: dict = Depends(require_role("ML_ADMIN")),
) -> AlertLogResponse:
    if not container.db_pool:
        return AlertLogResponse(alerts=[], total=0)

    conditions = []
    params = []
    param_idx = 1

    if alert_type:
        conditions.append(f"alert_type = ${param_idx}")
        params.append(alert_type)
        param_idx += 1

    if severity:
        conditions.append(f"severity = ${param_idx}")
        params.append(severity)
        param_idx += 1

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    async with container.db_pool.acquire() as conn:
        count_row = await conn.fetchrow(
            f"SELECT COUNT(*) as total FROM alert_log {where_clause}",
            *params,
        )

        rows = await conn.fetch(
            f"""
            SELECT * FROM alert_log {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """,
            *params,
            limit,
            offset,
        )

    alerts = [
        AlertLogEntry(
            alert_id=row["id"],
            alert_type=row["alert_type"],
            severity=row["severity"],
            channel=row["channel"],
            subject=row.get("subject"),
            entity_type=row.get("entity_type"),
            entity_id=row.get("entity_id"),
            delivered=row["delivered"],
            created_at=row["created_at"],
        )
        for row in rows
    ]

    return AlertLogResponse(alerts=alerts, total=count_row["total"])
