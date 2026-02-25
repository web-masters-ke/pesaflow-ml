"""AML Risk Scoring Service — Orchestrates sanctions screening, feature extraction, ML scoring, and case generation.

Supports maturity-aware scoring mode escalation:
  COLD    → Rule-based decisions only (skip ML model)
  WARMING → ML score with higher review thresholds + rule overrides weighted 2x
  WARM    → Standard ML scoring with confidence weighting
  HOT     → Ensemble scoring (stacked models)
"""

import time
import uuid
from typing import Any

import redis.asyncio as redis
from loguru import logger

from feature_engineering.aml_features import AMLFeatureExtractor
from models.aml.aml_model import AMLRiskModel
from monitoring.data_maturity import DataMaturityDetector, MaturityLevel
from monitoring.metrics import PesaflowMetrics
from serving.app.schemas.aml import AMLExplanationResponse, AMLScoreRequest, AMLScoreResponse, SanctionsScreenResult
from serving.app.schemas.common import Decision, FeatureContribution, RiskLevel
from serving.app.services.decision_engine import AMLDecisionEngine, AMLThresholdConfig
from serving.app.services.resilience import get_circuit_breaker, with_retry
from serving.app.services.sanctions_service import SanctionsScreeningService


class AMLScoringService:
    """End-to-end AML risk scoring orchestrator with maturity-aware escalation."""

    def __init__(
        self,
        model: AMLRiskModel,
        feature_extractor: AMLFeatureExtractor,
        decision_engine: AMLDecisionEngine,
        sanctions_service: SanctionsScreeningService,
        metrics: PesaflowMetrics,
        redis_client: redis.Redis,
        db_pool: Any = None,
        maturity_detector: DataMaturityDetector | None = None,
        anomaly_model: Any = None,
        ensemble_model: Any = None,
    ):
        self._model = model
        self._features = feature_extractor
        self._decision = decision_engine
        self._sanctions = sanctions_service
        self._metrics = metrics
        self._redis = redis_client
        self._db = db_pool
        self._maturity = maturity_detector
        self._anomaly_model = anomaly_model
        self._ensemble_model = ensemble_model

    async def score_transaction(self, request: AMLScoreRequest) -> AMLScoreResponse:
        """Score a single transaction for AML risk with maturity-aware escalation."""
        start = time.time()
        correlation_id = str(uuid.uuid4())

        try:
            # Get maturity level
            maturity = await self._get_maturity()

            # Step 1: Sanctions screening — always applies regardless of maturity
            sanctions_result = await self._screen_sanctions(request)

            # Step 2: Check blacklists
            blacklisted_user = self._sanctions.is_blacklisted_user(str(request.user_id))
            blacklisted_device = self._sanctions.is_blacklisted_device(request.device_id)

            # Step 3: Extract features
            feature_vector = await self._features.extract(request)
            features_array = feature_vector.to_array()

            # Step 4: Maturity-aware scoring
            risk_score, confidence = await self._score_by_maturity(features_array, maturity)

            # Step 5: Apply country risk weight
            if request.geo_location and request.geo_location.country.upper() in {"IR", "KP", "SY", "AF", "YE"}:
                risk_score = min(1.0, risk_score + 0.15)

            # Step 6: Detect structuring
            structuring_detected = feature_vector.structuring_score_24h > 0.5

            # Step 7: Detect velocity spike
            velocity_spike = await self._detect_velocity_spike(str(request.user_id), feature_vector.velocity_1h)

            # Step 8: Apply decision engine (confidence-aware)
            decision, risk_level, rule_overrides = self._decision.evaluate(
                score=risk_score,
                sanctions_match=sanctions_result.matched and sanctions_result.match_type in ("EXACT", "ALIAS"),
                blacklisted_user=blacklisted_user,
                blacklisted_device=blacklisted_device,
                watchlist_hit=sanctions_result.matched and sanctions_result.confidence >= 0.95,
                structuring_detected=structuring_detected,
                velocity_spike=velocity_spike,
                confidence=confidence,
                maturity=maturity,
            )

            latency_ms = int((time.time() - start) * 1000)

            # Step 9: Record metrics
            self._metrics.record_aml_scoring(
                latency_ms=latency_ms,
                decision=decision.value,
                risk_level=risk_level.value,
                score=risk_score,
            )

            # Step 10: Generate top risk factors
            top_risk_factors = self._compute_top_risk_factors(feature_vector, rule_overrides)

            # Step 10b: Inline SHAP for non-APPROVE decisions (MEDIUM+ risk)
            top_features = None
            if decision != Decision.APPROVE:
                top_features = self._compute_inline_shap(features_array)

            # Step 11: Store prediction (capture prediction_id)
            prediction_id = await self._store_prediction(
                request=request,
                risk_score=risk_score,
                risk_level=risk_level,
                decision=decision,
                feature_vector=feature_vector,
                top_risk_factors=top_risk_factors,
                latency_ms=latency_ms,
            )

            # Step 12: Generate case if needed (with prediction_id for label propagation)
            if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                await self._create_case(request, risk_score, risk_level, rule_overrides, prediction_id=prediction_id)

            # Step 13: Update velocity counters
            await self._features.update_velocity_counters(str(request.user_id), request.amount)

            return AMLScoreResponse(
                transaction_id=request.transaction_id,
                risk_score=round(risk_score, 4),
                risk_level=risk_level,
                decision=decision,
                model_version=self._model.version,
                threshold_version=self._decision.thresholds.version,
                top_risk_factors=top_risk_factors,
                rule_overrides=rule_overrides,
                top_features=top_features,
                latency_ms=latency_ms,
                correlation_id=correlation_id,
            )

        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            self._metrics.record_aml_error()
            logger.error(f"AML scoring failed for {request.transaction_id}: {e}")

            # Fail closed
            return AMLScoreResponse(
                transaction_id=request.transaction_id,
                risk_score=1.0,
                risk_level=RiskLevel.CRITICAL,
                decision=Decision.BLOCK,
                model_version=self._model.version,
                threshold_version=self._decision.thresholds.version,
                top_risk_factors=["MODEL_FAILURE"],
                rule_overrides=["FAIL_CLOSED"],
                latency_ms=latency_ms,
                correlation_id=correlation_id,
            )

    async def _score_by_maturity(self, features_array: list[float], maturity: MaturityLevel) -> tuple[float, float]:
        """Score based on data maturity level, returns (risk_score, confidence).

        Blending strategy — progressively trusts supervised ML as data matures:
          COLD    → 0% ML, 0% anomaly, 100% rules. Confidence=0.1.
                    Pure heuristic scoring from feature thresholds.
          WARMING → 30% ML + 70% anomaly (isolation forest). Confidence=0.3.
                    Anomaly model dominates because supervised signal is weak.
          WARM    → 70% ML + 30% anomaly. Confidence varies by model calibration.
                    Supervised model is primary, anomaly provides diversity.
          HOT     → 100% ensemble (stacked LGB+XGB+RF) or 100% ML. Confidence>=0.95.
                    Fully trusted supervised signal, no anomaly blending.
        """
        if maturity == MaturityLevel.COLD:
            score = self._rule_based_score(features_array)
            return score, 0.1

        if maturity == MaturityLevel.WARMING:
            ml_score = self._model.predict(features_array)
            anomaly_score = self._get_anomaly_score(features_array)
            blended = 0.3 * ml_score + 0.7 * anomaly_score
            return blended, 0.3

        if maturity == MaturityLevel.WARM:
            score, confidence = self._model.predict_with_confidence(features_array, maturity_confidence=0.7)
            anomaly_score = self._get_anomaly_score(features_array)
            blended = 0.7 * score + 0.3 * anomaly_score
            return blended, confidence

        # HOT: Use ensemble if available
        if self._ensemble_model and self._ensemble_model.is_loaded:
            score = self._ensemble_model.predict(features_array)
            return score, 0.95

        score, confidence = self._model.predict_with_confidence(features_array, maturity_confidence=1.0)
        return score, confidence

    def _rule_based_score(self, features: list[float]) -> float:
        """Compute heuristic AML risk score from features when ML is not available."""
        score = 0.2
        if len(features) >= 22:
            if features[1] > 10:  # velocity_1h
                score += 0.2
            if features[16] > 0:  # high_risk_country_flag
                score += 0.25
            if features[20] > 0.5:  # structuring_score_24h
                score += 0.2
            if features[21] > 0:  # rapid_drain_flag
                score += 0.15
            if features[14] > 0:  # circular_transfer_flag
                score += 0.15
            if features[17] > 0.3:  # sanctions_proximity_score
                score += 0.2
        return max(0.0, min(1.0, score))

    def _get_anomaly_score(self, features: list[float]) -> float:
        """Get anomaly score from isolation forest if available."""
        if self._anomaly_model and self._anomaly_model.is_loaded:
            try:
                return self._anomaly_model.predict(features)
            except Exception:
                pass
        return 0.5

    def _compute_inline_shap(self, features_array: list[float]) -> list[FeatureContribution] | None:
        """Compute SHAP values inline during scoring for non-APPROVE decisions."""
        try:
            shap_values = self._model.get_shap_values(features_array)
            return [
                FeatureContribution(feature=sv["feature"], value=sv["value"], impact=sv["impact"]) for sv in shap_values
            ]
        except Exception as e:
            logger.warning(f"Inline SHAP computation failed: {e}")
            return None

    async def _get_maturity(self) -> MaturityLevel:
        """Get current maturity level for the AML domain.

        Returns COLD when model artifacts are not loaded, ensuring the system
        uses rule-based scoring from day one instead of failing closed.
        """
        if not self._model.is_loaded:
            return MaturityLevel.COLD
        if self._maturity:
            return await self._maturity.get_cached_level("aml")
        return MaturityLevel.HOT

    async def explain_transaction(self, transaction_id: str) -> AMLExplanationResponse | None:
        """Get SHAP explanation for an AML-scored transaction."""
        if not self._db:
            return None

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM aml_predictions
                    WHERE transaction_id = $1
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    transaction_id,
                )

                if not row:
                    return None

                feature_snapshot = row.get("feature_snapshot", {})
                features_array = (
                    list(feature_snapshot.values()) if isinstance(feature_snapshot, dict) else feature_snapshot
                )

                shap_values = self._model.get_shap_values(features_array)
                top_features = [
                    FeatureContribution(feature=sv["feature"], value=sv["value"], impact=sv["impact"])
                    for sv in shap_values
                ]

                return AMLExplanationResponse(
                    transaction_id=row["transaction_id"],
                    risk_score=float(row["risk_score"]),
                    risk_level=RiskLevel(row["risk_level"]),
                    decision=Decision(row["decision"]),
                    top_features=top_features,
                    explanation_method="SHAP",
                    confidence=float(row["risk_score"]),
                    model_version=self._model.version,
                )
        except Exception as e:
            logger.error(f"AML explanation retrieval failed: {e}")
            return None

    async def _screen_sanctions(self, request: AMLScoreRequest) -> SanctionsScreenResult:
        """Screen transaction parties against sanctions lists."""
        # In production, would fetch user name from User Service
        country = request.geo_location.country if request.geo_location else ""
        result = await self._sanctions.screen_user(
            user_name="",  # Would come from user service
            user_id=str(request.user_id),
            country=country,
        )
        return result

    async def _detect_velocity_spike(self, user_id: str, current_velocity_1h: int) -> bool:
        """Detect sudden velocity spike vs historical baseline."""
        try:
            hist_key = f"aml:user:{user_id}:avg_velocity_1h"
            historical_avg = float(await self._redis.get(hist_key) or 0)
            if historical_avg > 0 and current_velocity_1h > historical_avg * 3:
                return True
        except Exception:
            pass
        return False

    def _compute_top_risk_factors(self, feature_vector: Any, rule_overrides: list[str]) -> list[str]:
        """Identify top risk factors for response."""
        factors: list[str] = list(rule_overrides)

        if feature_vector.velocity_1h > 5:
            factors.append("velocity_spike")
        if feature_vector.high_risk_country_flag:
            factors.append("high_risk_country")
        if feature_vector.new_device_flag:
            factors.append("new_device")
        if feature_vector.sanctions_proximity_score > 0.3:
            factors.append("sanctions_proximity")
        if feature_vector.structuring_score_24h > 0.3:
            factors.append("structuring_pattern")
        if feature_vector.network_risk_score > 0.5:
            factors.append("network_risk")
        if feature_vector.rapid_drain_flag:
            factors.append("rapid_drain")
        if feature_vector.ip_country_mismatch:
            factors.append("ip_country_mismatch")

        return factors[:10]

    @with_retry(max_attempts=3, backoff_base=0.1, backoff_max=2.0)
    async def _store_prediction(
        self,
        request: AMLScoreRequest,
        risk_score: float,
        risk_level: RiskLevel,
        decision: Decision,
        feature_vector: Any,
        top_risk_factors: list[str],
        latency_ms: int,
    ) -> str | None:
        """Persist AML prediction. Returns prediction_id for case linkage."""
        if not self._db:
            return None

        prediction_id = str(uuid.uuid4())
        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO aml_predictions
                    (id, transaction_id, user_id, model_id, risk_score, risk_level,
                     decision, top_risk_factors, feature_snapshot, threshold_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    prediction_id,
                    str(request.transaction_id),
                    str(request.user_id),
                    self._model.version,
                    risk_score,
                    risk_level.value,
                    decision.value,
                    top_risk_factors,
                    feature_vector.model_dump(),
                    self._decision.thresholds.version,
                )
            return prediction_id
        except Exception as e:
            logger.error(f"Failed to store AML prediction: {e}")
            return None

    @with_retry(max_attempts=3, backoff_base=0.1, backoff_max=2.0)
    async def _create_case(
        self,
        request: AMLScoreRequest,
        risk_score: float,
        risk_level: RiskLevel,
        triggers: list[str],
        prediction_id: str | None = None,
    ) -> None:
        """Auto-generate AML case for high-risk transactions."""
        if not self._db:
            return

        try:
            trigger_reason = triggers[0] if triggers else "HIGH_RISK_SCORE"
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO aml_cases
                    (id, entity_type, entity_id, trigger_reason, risk_score, priority, status, prediction_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    str(uuid.uuid4()),
                    "TRANSACTION",
                    str(request.transaction_id),
                    trigger_reason,
                    risk_score,
                    risk_level.value,
                    "OPEN",
                    prediction_id,
                )
                logger.info(f"AML case created for transaction {request.transaction_id}")
        except Exception as e:
            logger.error(f"Failed to create AML case: {e}")
