"""AML Risk Scoring API Routes."""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from serving.app.api.dependencies import get_aml_service
from serving.app.middleware.auth import verify_jwt
from serving.app.schemas.aml import (
    AMLBatchScoreRequest,
    AMLBatchScoreResponse,
    AMLExplanationResponse,
    AMLScoreRequest,
    AMLScoreResponse,
)
from serving.app.services.aml_scoring_service import AMLScoringService

router = APIRouter(prefix="/aml", tags=["AML Risk Scoring"])


@router.post(
    "/score",
    response_model=AMLScoreResponse,
    summary="Score transaction for AML risk",
    description="Real-time AML risk scoring with sanctions screening, structuring detection, and velocity monitoring.",
)
async def score_transaction(
    request: AMLScoreRequest,
    service: AMLScoringService = Depends(get_aml_service),
    _auth: dict = Depends(verify_jwt),
) -> AMLScoreResponse:
    return await service.score_transaction(request)


@router.post(
    "/score/batch",
    response_model=AMLBatchScoreResponse,
    summary="Batch score transactions for AML risk",
)
async def batch_score_transactions(
    request: AMLBatchScoreRequest,
    service: AMLScoringService = Depends(get_aml_service),
    _auth: dict = Depends(verify_jwt),
) -> AMLBatchScoreResponse:
    start = time.time()
    results = []
    failed = 0

    for tx in request.transactions:
        try:
            result = await service.score_transaction(tx)
            results.append(result)
        except Exception:
            failed += 1

    duration_ms = int((time.time() - start) * 1000)

    return AMLBatchScoreResponse(
        total=len(request.transactions),
        processed=len(results),
        failed=failed,
        results=results,
        model_version=results[0].model_version if results else "unknown",
        duration_ms=duration_ms,
    )


@router.get(
    "/explanation/{transaction_id}",
    response_model=AMLExplanationResponse,
    summary="Get AML risk explanation for a transaction",
    description="Returns SHAP-based feature contributions explaining the AML risk score.",
)
async def get_explanation(
    transaction_id: str,
    service: AMLScoringService = Depends(get_aml_service),
    _auth: dict = Depends(verify_jwt),
) -> AMLExplanationResponse:
    result = await service.explain_transaction(transaction_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction prediction not found")
    return result
