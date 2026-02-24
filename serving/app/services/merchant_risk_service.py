"""Merchant Risk Scoring Service — Orchestrates merchant risk assessment pipeline.

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

from feature_engineering.merchant_features import MerchantFeatureExtractor
from models.merchant.merchant_model import MerchantRiskModel
from monitoring.data_maturity import DataMaturityDetector, MaturityLevel
from monitoring.metrics import PesaflowMetrics
from serving.app.schemas.common import Decision, FeatureContribution, RiskLevel
from serving.app.schemas.merchant import (
    MerchantExplanationResponse,
    MerchantFeatureVector,
    MerchantScoreRequest,
    MerchantScoreResponse,
)
from serving.app.services.decision_engine import MerchantDecisionEngine, MerchantThresholdConfig
from serving.app.services.resilience import with_retry


class MerchantRiskService:
    """End-to-end merchant risk scoring orchestrator with maturity-aware escalation."""

    def __init__(
        self,
        model: MerchantRiskModel,
        feature_extractor: MerchantFeatureExtractor,
        decision_engine: MerchantDecisionEngine,
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
        self._metrics = metrics
        self._redis = redis_client
        self._db = db_pool
        self._maturity = maturity_detector
        self._anomaly_model = anomaly_model
        self._ensemble_model = ensemble_model

    async def score_merchant(self, request: MerchantScoreRequest) -> MerchantScoreResponse:
        """Score a merchant for risk with maturity-aware escalation."""
        start = time.time()
        correlation_id = str(uuid.uuid4())

        try:
            # Get maturity level
            maturity = await self._get_maturity()

            # Step 1: Extract features
            feature_vector = await self._features.extract(request)
            features_array = feature_vector.to_array()

            # Step 2: Maturity-aware scoring
            risk_score, confidence = await self._score_by_maturity(
                features_array, maturity, feature_vector
            )

            # Step 3: Apply decision engine (confidence-aware)
            decision, risk_level, merchant_tier, rule_overrides = self._decision.evaluate(
                score=risk_score,
                chargeback_rate=feature_vector.chargeback_rate_90d,
                fraud_rate=feature_vector.fraud_transaction_rate,
                velocity_spike=bool(feature_vector.velocity_spike_flag),
                confidence=confidence,
                maturity=maturity,
            )

            latency_ms = int((time.time() - start) * 1000)

            # Step 4: Record metrics
            self._metrics.record_merchant_scoring(
                latency_ms=latency_ms,
                decision=decision.value,
                risk_level=risk_level.value,
                score=risk_score,
                tier=merchant_tier,
            )

            # Step 5: Compute top risk factors
            top_risk_factors = self._compute_risk_factors(feature_vector, rule_overrides)

            # Step 5b: Inline SHAP for non-APPROVE decisions (MEDIUM+ risk)
            top_features = None
            if decision != Decision.APPROVE:
                top_features = self._compute_inline_shap(features_array)

            # Step 6: Store prediction
            await self._store_prediction(
                request=request,
                risk_score=risk_score,
                risk_level=risk_level,
                decision=decision,
                merchant_tier=merchant_tier,
                feature_vector=feature_vector,
                latency_ms=latency_ms,
            )

            # Step 7: Update velocity counters
            if request.customer_id:
                await self._features.update_velocity_counters(
                    str(request.merchant_id),
                    request.amount,
                    str(request.customer_id),
                )

            return MerchantScoreResponse(
                merchant_id=request.merchant_id,
                risk_score=round(risk_score, 4),
                risk_level=risk_level,
                decision=decision,
                merchant_tier=merchant_tier,
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
            self._metrics.record_merchant_error()
            logger.error(f"Merchant scoring failed for {request.merchant_id}: {e}")

            # Fail closed: block on model failure for consistency with fraud/AML services
            return MerchantScoreResponse(
                merchant_id=request.merchant_id,
                risk_score=1.0,
                risk_level=RiskLevel.CRITICAL,
                decision=Decision.BLOCK,
                merchant_tier="BLOCKED",
                model_version=self._model.version,
                threshold_version=self._decision.thresholds.version,
                top_risk_factors=["MODEL_FAILURE"],
                rule_overrides=["FAIL_CLOSED"],
                latency_ms=latency_ms,
                correlation_id=correlation_id,
            )

    async def _score_by_maturity(
        self,
        features_array: list[float],
        maturity: MaturityLevel,
        feature_vector: MerchantFeatureVector,
    ) -> tuple[float, float]:
        """Score based on data maturity level, returns (risk_score, confidence).

        Blending strategy — progressively trusts supervised ML as data matures:
          COLD    → 0% ML, 0% anomaly, 100% rules. Confidence=0.1.
                    Pure heuristic scoring from chargeback/fraud rates and feature thresholds.
          WARMING → 30% ML + 70% anomaly (isolation forest). Confidence=0.3.
                    Anomaly model dominates because supervised signal is weak.
          WARM    → 70% ML + 30% anomaly. Confidence varies by model calibration.
                    Supervised model is primary, anomaly provides diversity.
          HOT     → 100% ensemble (stacked LGB+XGB+RF) or 100% ML. Confidence>=0.95.
                    Fully trusted supervised signal, no anomaly blending.
        """
        if maturity == MaturityLevel.COLD:
            score = self._rule_based_score(feature_vector)
            return score, 0.1

        if maturity == MaturityLevel.WARMING:
            ml_score = self._model.predict(features_array)
            anomaly_score = self._get_anomaly_score(features_array)
            blended = 0.3 * ml_score + 0.7 * anomaly_score
            return blended, 0.3

        if maturity == MaturityLevel.WARM:
            score, confidence = self._model.predict_with_confidence(
                features_array, maturity_confidence=0.7
            )
            anomaly_score = self._get_anomaly_score(features_array)
            blended = 0.7 * score + 0.3 * anomaly_score
            return blended, confidence

        # HOT: Use ensemble if available
        if self._ensemble_model and self._ensemble_model.is_loaded:
            score = self._ensemble_model.predict(features_array)
            return score, 0.95

        score, confidence = self._model.predict_with_confidence(
            features_array, maturity_confidence=1.0
        )
        return score, confidence

    def _rule_based_score(self, fv: MerchantFeatureVector) -> float:
        """Compute heuristic merchant risk score from feature vector."""
        score = 0.2
        if fv.chargeback_rate_90d > 0.03:
            score += 0.3
        if fv.fraud_transaction_rate > 0.02:
            score += 0.25
        if fv.velocity_spike_flag:
            score += 0.15
        if fv.high_risk_customer_ratio > 0.15:
            score += 0.1
        if fv.account_age_days < 30:
            score += 0.1
        if fv.mcc_risk_score > 0.5:
            score += 0.1
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
                FeatureContribution(feature=sv["feature"], value=sv["value"], impact=sv["impact"])
                for sv in shap_values
            ]
        except Exception as e:
            logger.warning(f"Inline SHAP computation failed: {e}")
            return None

    async def _get_maturity(self) -> MaturityLevel:
        """Get current maturity level for the merchant domain."""
        if self._maturity:
            return await self._maturity.get_cached_level("merchant")
        return MaturityLevel.HOT

    async def explain_merchant(self, merchant_id: str) -> MerchantExplanationResponse | None:
        """Get SHAP explanation for a merchant risk score."""
        if not self._db:
            return None

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM merchant_risk_predictions
                    WHERE merchant_id = $1
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    merchant_id,
                )

                if not row:
                    return None

                feature_snapshot = row.get("feature_snapshot", {})
                features_array = list(feature_snapshot.values()) if isinstance(feature_snapshot, dict) else feature_snapshot

                shap_values = self._model.get_shap_values(features_array)
                top_features = [
                    FeatureContribution(feature=sv["feature"], value=sv["value"], impact=sv["impact"])
                    for sv in shap_values
                ]

                return MerchantExplanationResponse(
                    merchant_id=row["merchant_id"],
                    risk_score=float(row["risk_score"]),
                    risk_level=RiskLevel(row["risk_level"]),
                    decision=Decision(row["decision"]),
                    top_features=top_features,
                    explanation_method="SHAP",
                    confidence=float(row["risk_score"]),
                    model_version=self._model.version,
                )
        except Exception as e:
            logger.error(f"Merchant explanation retrieval failed: {e}")
            return None

    def _compute_risk_factors(self, fv: MerchantFeatureVector, overrides: list[str]) -> list[str]:
        """Identify top merchant risk factors."""
        factors: list[str] = list(overrides)

        if fv.chargeback_rate_90d > 0.03:
            factors.append("high_chargeback_rate")
        if fv.fraud_transaction_rate > 0.02:
            factors.append("high_fraud_rate")
        if fv.velocity_spike_flag:
            factors.append("velocity_spike")
        if fv.high_risk_customer_ratio > 0.15:
            factors.append("high_risk_customers")
        if fv.cross_border_ratio > 0.3:
            factors.append("high_cross_border_ratio")
        if fv.mcc_risk_score > 0.5:
            factors.append("high_risk_mcc")
        if fv.account_age_days < 30:
            factors.append("new_merchant")
        if fv.avg_customer_risk_score > 0.4:
            factors.append("risky_customer_base")

        return factors[:10]

    @with_retry(max_attempts=3, backoff_base=0.1, backoff_max=2.0)
    async def _store_prediction(
        self,
        request: MerchantScoreRequest,
        risk_score: float,
        risk_level: RiskLevel,
        decision: Decision,
        merchant_tier: str,
        feature_vector: MerchantFeatureVector,
        latency_ms: int,
    ) -> None:
        """Persist merchant risk prediction."""
        if not self._db:
            return

        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO merchant_risk_predictions
                    (id, merchant_id, risk_score, risk_level, decision, merchant_tier,
                     model_version, feature_snapshot, latency_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    str(uuid.uuid4()),
                    str(request.merchant_id),
                    risk_score,
                    risk_level.value,
                    decision.value,
                    merchant_tier,
                    self._model.version,
                    feature_vector.model_dump(),
                    latency_ms,
                )
        except Exception as e:
            logger.error(f"Failed to store merchant prediction: {e}")
