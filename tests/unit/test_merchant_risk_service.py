"""Unit tests for Merchant Risk Scoring Service."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from monitoring.data_maturity import MaturityLevel
from serving.app.schemas.common import Decision, RiskLevel
from serving.app.schemas.merchant import MerchantFeatureVector, MerchantScoreRequest
from serving.app.services.merchant_risk_service import MerchantRiskService


def _make_request(**kwargs):
    defaults = {
        "merchant_id": uuid.uuid4(),
        "amount": 10000.0,
        "currency": "KES",
        "timestamp": datetime(2024, 6, 15, 14, 30),
    }
    defaults.update(kwargs)
    return MerchantScoreRequest(**defaults)


def _make_feature_vector(**kwargs):
    return MerchantFeatureVector(**kwargs)


def _build_service(
    maturity=MaturityLevel.HOT,
    predict_score=0.2,
    predict_confidence=0.9,
    anomaly_score=0.5,
    ensemble_score=None,
    feature_overrides=None,
):
    """Build a MerchantRiskService with mocked dependencies."""
    model = MagicMock()
    model.version = "v1.0"
    model.predict.return_value = predict_score
    model.predict_with_confidence.return_value = (predict_score, predict_confidence)
    model.get_shap_values.return_value = [
        {"feature": "chargeback_rate_90d", "value": 0.01, "impact": 0.3},
        {"feature": "fraud_transaction_rate", "value": 0.005, "impact": 0.2},
    ]

    fv = _make_feature_vector(**(feature_overrides or {}))
    feature_extractor = MagicMock()
    feature_extractor.extract = AsyncMock(return_value=fv)
    feature_extractor.update_velocity_counters = AsyncMock()

    from serving.app.services.decision_engine import MerchantDecisionEngine, MerchantThresholdConfig

    decision_engine = MerchantDecisionEngine(MerchantThresholdConfig())

    metrics = MagicMock()

    redis_client = AsyncMock()

    maturity_detector = MagicMock()
    maturity_detector.get_cached_level = AsyncMock(return_value=maturity)

    anomaly_model = MagicMock()
    anomaly_model.is_loaded = True
    anomaly_model.predict.return_value = anomaly_score

    ensemble_model = None
    if ensemble_score is not None:
        ensemble_model = MagicMock()
        ensemble_model.is_loaded = True
        ensemble_model.predict.return_value = ensemble_score

    return MerchantRiskService(
        model=model,
        feature_extractor=feature_extractor,
        decision_engine=decision_engine,
        metrics=metrics,
        redis_client=redis_client,
        db_pool=None,
        maturity_detector=maturity_detector,
        anomaly_model=anomaly_model,
        ensemble_model=ensemble_model,
    )


class TestMerchantRiskServiceMaturity:
    @pytest.mark.asyncio
    async def test_hot_maturity_ensemble(self):
        service = _build_service(maturity=MaturityLevel.HOT, ensemble_score=0.2)
        resp = await service.score_merchant(_make_request())
        assert resp.decision == Decision.APPROVE
        assert resp.merchant_tier == "STANDARD"

    @pytest.mark.asyncio
    async def test_cold_maturity_rule_based(self):
        service = _build_service(maturity=MaturityLevel.COLD)
        resp = await service.score_merchant(_make_request())
        assert 0.0 <= resp.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_warming_maturity_blended(self):
        service = _build_service(maturity=MaturityLevel.WARMING, predict_score=0.4, anomaly_score=0.6)
        resp = await service.score_merchant(_make_request())
        assert 0.0 <= resp.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_warm_maturity_ml_primary(self):
        service = _build_service(maturity=MaturityLevel.WARM, predict_score=0.3, anomaly_score=0.5)
        resp = await service.score_merchant(_make_request())
        assert 0.0 <= resp.risk_score <= 1.0


class TestMerchantChargebackOverride:
    @pytest.mark.asyncio
    async def test_high_chargeback_rate_blocks(self):
        service = _build_service(
            predict_score=0.3,
            feature_overrides={"chargeback_rate_90d": 0.15},
        )
        resp = await service.score_merchant(_make_request())
        assert resp.decision == Decision.BLOCK
        assert "EXCESSIVE_CHARGEBACKS" in resp.rule_overrides

    @pytest.mark.asyncio
    async def test_high_fraud_rate_restricts(self):
        service = _build_service(
            predict_score=0.3,
            feature_overrides={"fraud_transaction_rate": 0.06},
        )
        resp = await service.score_merchant(_make_request())
        assert resp.decision == Decision.REVIEW
        assert "HIGH_FRAUD_RATE" in resp.rule_overrides


class TestMerchantFailClosed:
    @pytest.mark.asyncio
    async def test_fail_closed_on_exception(self):
        service = _build_service()
        service._features.extract = AsyncMock(side_effect=RuntimeError("DB down"))
        resp = await service.score_merchant(_make_request())
        assert resp.decision == Decision.BLOCK
        assert resp.risk_level == RiskLevel.CRITICAL
        assert resp.merchant_tier == "BLOCKED"
        assert "FAIL_CLOSED" in resp.rule_overrides


class TestMerchantSHAP:
    @pytest.mark.asyncio
    async def test_shap_included_for_high_risk(self):
        service = _build_service(predict_score=0.95, predict_confidence=0.95)
        resp = await service.score_merchant(_make_request())
        if resp.decision != Decision.APPROVE:
            assert resp.top_features is not None
