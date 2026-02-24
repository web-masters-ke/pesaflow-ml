"""Unit tests for Fraud Scoring Service."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monitoring.data_maturity import MaturityLevel
from serving.app.schemas.common import Decision, RiskLevel
from serving.app.schemas.fraud import FraudFeatureVector, FraudScoreRequest, OverrideFlags
from serving.app.services.fraud_scoring_service import FraudScoringService


def _make_request(**kwargs):
    defaults = {
        "transaction_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "amount": 5000.0,
        "currency": "KES",
        "transaction_type": "WALLET_TRANSFER",
        "timestamp": datetime(2024, 6, 15, 14, 30),
    }
    defaults.update(kwargs)
    return FraudScoreRequest(**defaults)


def _make_feature_vector(**kwargs):
    return FraudFeatureVector(amount=5000.0, **kwargs)


def _build_service(
    maturity=MaturityLevel.HOT,
    predict_score=0.3,
    predict_confidence=0.9,
    overrides=None,
    anomaly_score=0.5,
    ensemble_score=None,
    shap_values=None,
):
    """Build a FraudScoringService with mocked dependencies."""
    model = MagicMock()
    model.version = "v1.0"
    model.predict.return_value = predict_score
    model.predict_with_confidence.return_value = (predict_score, predict_confidence)
    model.compute_feature_hash.return_value = "hash123"
    model.get_shap_values.return_value = shap_values or [
        {"feature": "amount", "value": 5000.0, "impact": 0.3},
        {"feature": "velocity_1h", "value": 5, "impact": 0.2},
    ]

    feature_extractor = MagicMock()
    fv = _make_feature_vector()
    feature_extractor.extract = AsyncMock(return_value=fv)
    feature_extractor.update_velocity_counters = AsyncMock()

    from serving.app.services.decision_engine import FraudDecisionEngine, ThresholdConfig

    decision_engine = FraudDecisionEngine(
        ThresholdConfig(approve_below=0.60, review_above=0.75, block_above=0.90)
    )

    sanctions_service = MagicMock()
    sanctions_service.is_blacklisted_device.return_value = False
    sanctions_service.is_blacklisted_ip.return_value = False
    sanctions_service.is_blacklisted_user.return_value = False

    if overrides:
        sanctions_service.is_blacklisted_device.return_value = overrides.get("device", False)
        sanctions_service.is_blacklisted_ip.return_value = overrides.get("ip", False)
        sanctions_service.is_blacklisted_user.return_value = overrides.get("user", False)

    metrics = MagicMock()

    redis_client = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.pipeline.return_value = AsyncMock()

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

    return FraudScoringService(
        model=model,
        feature_extractor=feature_extractor,
        decision_engine=decision_engine,
        sanctions_service=sanctions_service,
        metrics=metrics,
        redis_client=redis_client,
        db_pool=None,
        maturity_detector=maturity_detector,
        anomaly_model=anomaly_model,
        ensemble_model=ensemble_model,
    )


class TestFraudScoringServiceMaturity:
    @pytest.mark.asyncio
    async def test_hot_maturity_uses_ensemble(self):
        service = _build_service(maturity=MaturityLevel.HOT, ensemble_score=0.2)
        req = _make_request()
        resp = await service.score_transaction(req)
        assert resp.decision == Decision.APPROVE
        assert resp.risk_score <= 0.6

    @pytest.mark.asyncio
    async def test_cold_maturity_rule_based(self):
        service = _build_service(maturity=MaturityLevel.COLD)
        req = _make_request()
        resp = await service.score_transaction(req)
        # Cold uses rule-based, should get a score
        assert 0.0 <= resp.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_warming_maturity_blended(self):
        service = _build_service(maturity=MaturityLevel.WARMING, predict_score=0.4, anomaly_score=0.6)
        req = _make_request()
        resp = await service.score_transaction(req)
        # Blended: 0.3*0.4 + 0.7*0.6 = 0.54
        assert 0.0 <= resp.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_warm_maturity_ml_primary(self):
        service = _build_service(maturity=MaturityLevel.WARM, predict_score=0.3, anomaly_score=0.5)
        req = _make_request()
        resp = await service.score_transaction(req)
        assert 0.0 <= resp.risk_score <= 1.0


class TestFraudScoringServiceFailClosed:
    @pytest.mark.asyncio
    async def test_fail_closed_on_exception(self):
        service = _build_service()
        service._features.extract = AsyncMock(side_effect=RuntimeError("DB down"))
        req = _make_request()
        resp = await service.score_transaction(req)
        assert resp.decision == Decision.BLOCK
        assert resp.risk_level == RiskLevel.CRITICAL
        assert "MODEL_FAILURE_FAIL_CLOSED" in resp.rule_overrides


class TestFraudScoringServiceOverrides:
    @pytest.mark.asyncio
    async def test_blacklisted_device_blocks(self):
        service = _build_service(overrides={"device": True})
        req = _make_request()
        resp = await service.score_transaction(req)
        assert resp.decision == Decision.BLOCK
        assert "BLACKLISTED_DEVICE" in resp.rule_overrides

    @pytest.mark.asyncio
    async def test_blacklisted_user_blocks(self):
        service = _build_service(overrides={"user": True})
        req = _make_request()
        resp = await service.score_transaction(req)
        assert resp.decision == Decision.BLOCK

    @pytest.mark.asyncio
    async def test_blacklisted_ip_blocks(self):
        service = _build_service(overrides={"ip": True})
        req = _make_request()
        resp = await service.score_transaction(req)
        assert resp.decision == Decision.BLOCK


class TestFraudScoringServiceSHAP:
    @pytest.mark.asyncio
    async def test_shap_included_for_non_approve(self):
        # High score => BLOCK => SHAP should be computed
        service = _build_service(maturity=MaturityLevel.HOT, predict_score=0.95, predict_confidence=0.95)
        req = _make_request()
        resp = await service.score_transaction(req)
        assert resp.decision != Decision.APPROVE
        assert resp.top_features is not None
        assert len(resp.top_features) > 0

    @pytest.mark.asyncio
    async def test_shap_not_included_for_approve(self):
        service = _build_service(maturity=MaturityLevel.HOT, predict_score=0.1, predict_confidence=0.95)
        req = _make_request()
        resp = await service.score_transaction(req)
        assert resp.decision == Decision.APPROVE
        assert resp.top_features is None


class TestVelocityCounterUpdates:
    @pytest.mark.asyncio
    async def test_velocity_counters_updated_on_success(self):
        service = _build_service()
        req = _make_request()
        await service.score_transaction(req)
        service._features.update_velocity_counters.assert_called_once()
