"""Merchant Risk Scoring Routes — REST API for merchant risk assessment."""

from fastapi import APIRouter, Depends, HTTPException, status

from serving.app.api.dependencies import get_merchant_service
from serving.app.middleware.auth import require_role
from serving.app.schemas.merchant import (
    MerchantBatchScoreRequest,
    MerchantBatchScoreResponse,
    MerchantExplanationResponse,
    MerchantScoreRequest,
    MerchantScoreResponse,
)
from serving.app.services.merchant_risk_service import MerchantRiskService

router = APIRouter(prefix="/merchant", tags=["Merchant Risk"])


@router.post("/score", response_model=MerchantScoreResponse, summary="Score merchant risk")
async def score_merchant(
    request: MerchantScoreRequest,
    service: MerchantRiskService = Depends(get_merchant_service),
    _auth: dict = Depends(require_role("ML_ADMIN", "FRAUD_ANALYST")),
) -> MerchantScoreResponse:
    return await service.score_merchant(request)


@router.post("/score/batch", response_model=MerchantBatchScoreResponse, summary="Batch score merchants")
async def batch_score_merchants(
    request: MerchantBatchScoreRequest,
    service: MerchantRiskService = Depends(get_merchant_service),
    _auth: dict = Depends(require_role("ML_ADMIN", "FRAUD_ANALYST")),
) -> MerchantBatchScoreResponse:
    import time

    start = time.time()
    results = []
    failed = 0

    for merchant_req in request.merchants:
        try:
            result = await service.score_merchant(merchant_req)
            results.append(result)
        except Exception:
            failed += 1

    duration_ms = int((time.time() - start) * 1000)

    return MerchantBatchScoreResponse(
        total=len(request.merchants),
        processed=len(results),
        failed=failed,
        results=results,
        model_version=results[0].model_version if results else "unknown",
        duration_ms=duration_ms,
    )


@router.get(
    "/explanation/{merchant_id}",
    response_model=MerchantExplanationResponse,
    summary="Get merchant risk explanation",
)
async def get_merchant_explanation(
    merchant_id: str,
    service: MerchantRiskService = Depends(get_merchant_service),
    _auth: dict = Depends(require_role("ML_ADMIN", "FRAUD_ANALYST")),
) -> MerchantExplanationResponse:
    explanation = await service.explain_merchant(merchant_id)
    if not explanation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No merchant risk data found for {merchant_id}",
        )
    return explanation
