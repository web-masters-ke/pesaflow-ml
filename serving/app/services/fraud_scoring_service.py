"""Fraud Scoring Service — Orchestrates feature extraction, ML scoring, decision, and audit.

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

from feature_engineering.fraud_features import FraudFeatureExtractor
from models.fraud.fraud_model import FraudDetectionModel
from monitoring.data_maturity import DataMaturityDetector, MaturityLevel
from monitoring.metrics import PesaflowMetrics
from serving.app.schemas.common import Decision, FeatureContribution, RiskLevel
from serving.app.schemas.fraud import (
    FraudExplanationResponse,
    FraudFeatureVector,
    FraudScoreRequest,
    FraudScoreResponse,
    OverrideFlags,
)
from serving.app.services.decision_engine import FraudDecisionEngine, ThresholdConfig
from serving.app.services.resilience import get_circuit_breaker, with_retry
from serving.app.services.sanctions_service import SanctionsScreeningService


class FraudScoringService:
    """End-to-end fraud scoring orchestrator with maturity-aware mode escalation."""

    def __init__(
        self,
        model: FraudDetectionModel,
        feature_extractor: FraudFeatureExtractor,
        decision_engine: FraudDecisionEngine,
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

    async def score_transaction(self, request: FraudScoreRequest) -> FraudScoreResponse:
        """Score a single transaction for fraud risk with maturity-aware escalation."""
        start = time.time()
        correlation_id = str(uuid.uuid4())

        try:
            # Get maturity level
            maturity = await self._get_maturity()

            # Step 1: Check hard overrides (blacklists) — always applies
            overrides = await self._check_overrides(request)

            # Step 2: Extract features
            feature_vector = await self._features.extract(request)
            features_array = feature_vector.to_array()

            # Step 3: Maturity-aware scoring
            risk_score, confidence = await self._score_by_maturity(
                features_array, maturity, overrides
            )

            # Step 4: Apply decision engine (confidence-aware)
            decision, risk_level, rule_overrides = self._decision.evaluate(
                risk_score, overrides, confidence=confidence, maturity=maturity
            )

            latency_ms = int((time.time() - start) * 1000)

            # Step 5: Record metrics
            self._metrics.record_fraud_scoring(
                latency_ms=latency_ms,
                decision=decision.value,
                risk_level=risk_level.value,
                score=risk_score,
            )

            # Step 6: Inline SHAP for non-APPROVE decisions (MEDIUM+ risk)
            top_features = None
            if decision != Decision.APPROVE:
                top_features = self._compute_inline_shap(features_array)

            # Step 7: Store prediction
            await self._store_prediction(
                request=request,
                risk_score=risk_score,
                risk_level=risk_level,
                decision=decision,
                feature_vector=feature_vector,
                latency_ms=latency_ms,
                correlation_id=correlation_id,
            )

            # Step 8: Update velocity counters
            await self._features.update_velocity_counters(str(request.user_id), request.amount)

            return FraudScoreResponse(
                transaction_id=request.transaction_id,
                risk_score=round(risk_score, 4),
                risk_level=risk_level,
                decision=decision,
                model_version=self._model.version,
                threshold_version=self._decision.thresholds.version,
                latency_ms=latency_ms,
                rule_overrides=rule_overrides,
                top_features=top_features,
                correlation_id=correlation_id,
            )

        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            self._metrics.record_fraud_error()
            logger.error(f"Fraud scoring failed for {request.transaction_id}: {e}")

            # Fail closed: block on model failure
            return FraudScoreResponse(
                transaction_id=request.transaction_id,
                risk_score=1.0,
                risk_level=RiskLevel.CRITICAL,
                decision=Decision.BLOCK,
                model_version=self._model.version,
                threshold_version=self._decision.thresholds.version,
                latency_ms=latency_ms,
                rule_overrides=["MODEL_FAILURE_FAIL_CLOSED"],
                correlation_id=correlation_id,
            )

    async def _score_by_maturity(
        self,
        features_array: list[float],
        maturity: MaturityLevel,
        overrides: OverrideFlags,
    ) -> tuple[float, float]:
        """Score based on data maturity level, returns (risk_score, confidence).

        Blending strategy — progressively trusts supervised ML as data matures:
          COLD    → 0% ML, 0% anomaly, 100% rules. Confidence=0.1.
                    Pure heuristic scoring from override flags and feature thresholds.
          WARMING → 30% ML + 70% anomaly (isolation forest). Confidence=0.3.
                    Anomaly model dominates because supervised signal is weak.
          WARM    → 70% ML + 30% anomaly. Confidence varies by model calibration.
                    Supervised model is primary, anomaly provides diversity.
          HOT     → 100% ensemble (stacked LGB+XGB+RF) or 100% ML. Confidence>=0.95.
                    Fully trusted supervised signal, no anomaly blending.
        """
        if maturity == MaturityLevel.COLD:
            # Pure rule-based: derive score from overrides/heuristics
            score = self._rule_based_score(features_array, overrides)
            return score, 0.1

        if maturity == MaturityLevel.WARMING:
            # ML + anomaly blend with conservative confidence
            ml_score = self._model.predict(features_array)
            anomaly_score = self._get_anomaly_score(features_array)
            alpha = 0.3  # Low trust in supervised model
            blended = alpha * ml_score + (1 - alpha) * anomaly_score
            return blended, 0.3

        if maturity == MaturityLevel.WARM:
            # Standard ML with confidence
            score, confidence = self._model.predict_with_confidence(
                features_array,
                maturity_confidence=0.7,
            )
            # Blend with anomaly signal
            anomaly_score = self._get_anomaly_score(features_array)
            blended = 0.7 * score + 0.3 * anomaly_score
            return blended, confidence

        # HOT: Use ensemble if available
        if self._ensemble_model and self._ensemble_model.is_loaded:
            score = self._ensemble_model.predict(features_array)
            return score, 0.95

        # Fallback to standard model
        score, confidence = self._model.predict_with_confidence(
            features_array,
            maturity_confidence=1.0,
        )
        return score, confidence

    def _rule_based_score(self, features: list[float], overrides: OverrideFlags) -> float:
        """Compute a heuristic risk score from rules when ML is not available."""
        score = 0.3  # Base risk

        # Override signals
        if overrides.velocity_anomaly:
            score += 0.3
        if overrides.new_account_high_amount:
            score += 0.2

        # Feature-based heuristics (indices match FRAUD_FEATURE_SCHEMA)
        if len(features) >= 13:
            # High amount relative to average
            avg_7d = features[0] if features[0] > 0 else 50.0
            amount = features[9]
            if amount > avg_7d * 5:
                score += 0.2

            # High velocity
            if features[1] > 10:  # transaction_velocity_1h
                score += 0.15

            # Historical fraud flag
            if features[5] > 0:
                score += 0.2

            # High device risk
            if features[6] > 0.5:
                score += 0.1

        return max(0.0, min(1.0, score))

    def _get_anomaly_score(self, features: list[float]) -> float:
        """Get anomaly score from isolation forest if available."""
        if self._anomaly_model and self._anomaly_model.is_loaded:
            try:
                return self._anomaly_model.predict(features)
            except Exception:
                pass
        return 0.5  # Neutral when unavailable

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
        """Get current maturity level for the fraud domain."""
        if self._maturity:
            return await self._maturity.get_cached_level("fraud")
        return MaturityLevel.HOT  # Default to full ML when no detector

    async def explain_transaction(self, transaction_id: str) -> FraudExplanationResponse | None:
        """Get SHAP explanation for a scored transaction."""
        # Retrieve stored prediction and feature snapshot
        if not self._db:
            return None

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT p.*, fs.feature_data, fs.feature_hash
                    FROM ml_predictions p
                    JOIN ml_prediction_feature_snapshot fs ON fs.prediction_id = p.id
                    WHERE p.transaction_id = $1
                    ORDER BY p.created_at DESC LIMIT 1
                    """,
                    transaction_id,
                )

                if not row:
                    return None

                feature_data = row["feature_data"]
                features_array = list(feature_data.values()) if isinstance(feature_data, dict) else feature_data

                shap_values = self._model.get_shap_values(features_array)
                top_features = [
                    FeatureContribution(feature=sv["feature"], value=sv["value"], impact=sv["impact"])
                    for sv in shap_values
                ]

                return FraudExplanationResponse(
                    transaction_id=row["transaction_id"],
                    risk_score=float(row["risk_score"]),
                    risk_level=RiskLevel(row["risk_level"]),
                    decision=Decision(row["decision"]),
                    top_features=top_features,
                    explanation_method="SHAP",
                    confidence=float(row["risk_score"]),
                    model_version=self._model.version,
                    feature_snapshot_hash=row.get("feature_hash"),
                )
        except Exception as e:
            logger.error(f"Explanation retrieval failed: {e}")
            return None

    async def _check_overrides(self, request: FraudScoreRequest) -> OverrideFlags:
        """Check blacklists and hard rules."""
        return OverrideFlags(
            blacklisted_device=self._sanctions.is_blacklisted_device(request.device_fingerprint),
            blacklisted_ip=self._sanctions.is_blacklisted_ip(request.ip_address),
            blacklisted_user=self._sanctions.is_blacklisted_user(str(request.user_id)),
            sanctioned_country=request.geo_location.country.upper() in {"IR", "KP", "SY", "CU"} if request.geo_location else False,
            velocity_anomaly=await self._check_velocity_anomaly(str(request.user_id)),
            new_account_high_amount=False,  # Would check account age vs amount
        )

    async def _check_velocity_anomaly(self, user_id: str) -> bool:
        """Check if transaction velocity exceeds threshold."""
        try:
            count_1h = int(await self._redis.get(f"user:{user_id}:txn_count:1h") or 0)
            return count_1h > 20  # Configurable threshold
        except Exception:
            return False

    @with_retry(max_attempts=3, backoff_base=0.1, backoff_max=2.0)
    async def _store_prediction(
        self,
        request: FraudScoreRequest,
        risk_score: float,
        risk_level: RiskLevel,
        decision: Decision,
        feature_vector: FraudFeatureVector,
        latency_ms: int,
        correlation_id: str,
    ) -> None:
        """Persist prediction and feature snapshot for audit."""
        if not self._db:
            return

        prediction_id = str(uuid.uuid4())
        feature_hash = self._model.compute_feature_hash(feature_vector.to_array())

        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO ml_predictions
                    (id, transaction_id, user_id, model_version_id, threshold_version_id,
                     risk_score, risk_level, decision, latency_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    prediction_id,
                    str(request.transaction_id),
                    str(request.user_id),
                    self._model.version,
                    self._decision.thresholds.version,
                    risk_score,
                    risk_level.value,
                    decision.value,
                    latency_ms,
                )

                await conn.execute(
                    """
                    INSERT INTO ml_prediction_feature_snapshot
                    (prediction_id, feature_data, feature_hash)
                    VALUES ($1, $2, $3)
                    """,
                    prediction_id,
                    feature_vector.model_dump(),
                    feature_hash,
                )
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
