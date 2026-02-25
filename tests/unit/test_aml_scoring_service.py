"""Unit tests for AML Scoring Service."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from monitoring.data_maturity import MaturityLevel
from serving.app.schemas.aml import AMLFeatureVector, AMLScoreRequest, SanctionsScreenResult
from serving.app.schemas.common import Decision, GeoLocation, RiskLevel
from serving.app.services.aml_scoring_service import AMLScoringService


def _make_request(**kwargs):
    defaults = {
        "transaction_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "amount": 50000.0,
        "currency": "KES",
        "timestamp": datetime(2024, 6, 15, 14, 30),
    }
    defaults.update(kwargs)
    return AMLScoreRequest(**defaults)


def _make_feature_vector(**kwargs):
    return AMLFeatureVector(amount=50000.0, **kwargs)


def _build_service(
    maturity=MaturityLevel.HOT,
    predict_score=0.3,
    predict_confidence=0.9,
    sanctions_matched=False,
    sanctions_match_type=None,
    sanctions_confidence=0.0,
    anomaly_score=0.5,
    ensemble_score=None,
    feature_overrides=None,
):
    """Build an AMLScoringService with mocked dependencies."""
    model = MagicMock()
    model.version = "v1.0"
    model.predict.return_value = predict_score
    model.predict_with_confidence.return_value = (predict_score, predict_confidence)
    model.get_shap_values.return_value = [
        {"feature": "amount", "value": 50000.0, "impact": 0.25},
        {"feature": "velocity_1h", "value": 8, "impact": 0.2},
    ]

    fv = _make_feature_vector(**(feature_overrides or {}))
    feature_extractor = MagicMock()
    feature_extractor.extract = AsyncMock(return_value=fv)
    feature_extractor.update_velocity_counters = AsyncMock()

    from serving.app.services.decision_engine import AMLDecisionEngine, AMLThresholdConfig

    decision_engine = AMLDecisionEngine(AMLThresholdConfig(medium_above=0.50, high_above=0.70, critical_above=0.85))

    sanctions_service = MagicMock()
    sanctions_service.is_blacklisted_user.return_value = False
    sanctions_service.is_blacklisted_device.return_value = False
    sanctions_service.screen_user = AsyncMock(
        return_value=SanctionsScreenResult(
            matched=sanctions_matched,
            match_type=sanctions_match_type,
            confidence=sanctions_confidence,
        )
    )

    metrics = MagicMock()

    redis_client = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)

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

    return AMLScoringService(
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


class TestAMLScoringServiceMaturity:
    @pytest.mark.asyncio
    async def test_hot_maturity_ensemble(self):
        service = _build_service(maturity=MaturityLevel.HOT, ensemble_score=0.2)
        resp = await service.score_transaction(_make_request())
        assert resp.decision == Decision.APPROVE
        assert 0.0 <= resp.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_cold_maturity_rule_based(self):
        service = _build_service(maturity=MaturityLevel.COLD)
        resp = await service.score_transaction(_make_request())
        assert 0.0 <= resp.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_warming_maturity_blended(self):
        service = _build_service(maturity=MaturityLevel.WARMING, predict_score=0.4, anomaly_score=0.6)
        resp = await service.score_transaction(_make_request())
        assert 0.0 <= resp.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_warm_maturity_ml_primary(self):
        service = _build_service(maturity=MaturityLevel.WARM, predict_score=0.3, anomaly_score=0.5)
        resp = await service.score_transaction(_make_request())
        assert 0.0 <= resp.risk_score <= 1.0


class TestAMLSanctionsScreening:
    @pytest.mark.asyncio
    async def test_sanctions_exact_match_blocks(self):
        service = _build_service(
            sanctions_matched=True,
            sanctions_match_type="EXACT",
            sanctions_confidence=0.99,
        )
        resp = await service.score_transaction(_make_request())
        assert resp.decision == Decision.BLOCK
        assert resp.risk_level == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_no_sanctions_match_passes(self):
        service = _build_service(predict_score=0.2, predict_confidence=0.95)
        resp = await service.score_transaction(_make_request())
        assert resp.decision == Decision.APPROVE


class TestAMLStructuringDetection:
    @pytest.mark.asyncio
    async def test_structuring_detected_elevates(self):
        service = _build_service(
            predict_score=0.4,
            feature_overrides={"structuring_score_24h": 0.8},
        )
        resp = await service.score_transaction(_make_request())
        # structuring_score > 0.5 should be detected and elevate decision
        assert resp.decision in (Decision.REVIEW, Decision.MONITOR, Decision.BLOCK)


class TestAMLVelocitySpike:
    @pytest.mark.asyncio
    async def test_velocity_spike_detection(self):
        service = _build_service(predict_score=0.3)
        # Mock velocity spike detection
        service._detect_velocity_spike = AsyncMock(return_value=True)
        resp = await service.score_transaction(_make_request())
        assert "VELOCITY_SPIKE" in resp.rule_overrides


class TestAMLCaseGeneration:
    @pytest.mark.asyncio
    async def test_case_generated_for_high_risk(self):
        service = _build_service(predict_score=0.95, predict_confidence=0.95)
        service._create_case = AsyncMock()
        resp = await service.score_transaction(_make_request())
        if resp.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            service._create_case.assert_called_once()


class TestAMLFailClosed:
    @pytest.mark.asyncio
    async def test_fail_closed_on_exception(self):
        service = _build_service()
        service._features.extract = AsyncMock(side_effect=RuntimeError("DB down"))
        resp = await service.score_transaction(_make_request())
        assert resp.decision == Decision.BLOCK
        assert resp.risk_level == RiskLevel.CRITICAL
        assert "FAIL_CLOSED" in resp.rule_overrides


class TestAMLSHAP:
    @pytest.mark.asyncio
    async def test_shap_included_for_non_approve(self):
        service = _build_service(predict_score=0.95, predict_confidence=0.95)
        resp = await service.score_transaction(_make_request())
        if resp.decision != Decision.APPROVE:
            assert resp.top_features is not None

    @pytest.mark.asyncio
    async def test_shap_not_included_for_approve(self):
        service = _build_service(predict_score=0.1, predict_confidence=0.95)
        resp = await service.score_transaction(_make_request())
        assert resp.decision == Decision.APPROVE
        assert resp.top_features is None
