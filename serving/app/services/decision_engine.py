"""Decision Engine — Converts ML scores into enforceable actions with rule overrides.

Supports confidence-aware decisions and maturity-based mode escalation.
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from monitoring.data_maturity import MaturityLevel
from serving.app.schemas.common import Decision, RiskLevel
from serving.app.schemas.fraud import OverrideFlags


@dataclass
class ThresholdConfig:
    """Configurable risk thresholds."""

    approve_below: float
    review_above: float
    block_above: float
    version: str = "tv1.0"

    def validate(self) -> bool:
        return self.approve_below < self.review_above < self.block_above


class FraudDecisionEngine:
    """Fraud decision engine — applies thresholds + override rules.

    Supports confidence-aware decisions: when confidence is low and maturity
    is COLD/WARMING, stricter thresholds are applied.
    """

    def __init__(self, thresholds: ThresholdConfig):
        self.thresholds = thresholds
        if not thresholds.validate():
            raise ValueError("Invalid thresholds: must be approve < review < block")

    def evaluate(
        self,
        score: float,
        overrides: OverrideFlags,
        confidence: float = 1.0,
        maturity: MaturityLevel = MaturityLevel.HOT,
    ) -> tuple[Decision, RiskLevel, list[str]]:
        """Evaluate transaction and return (decision, risk_level, override_reasons).

        Args:
            score: ML risk score (0.0-1.0)
            overrides: Hard rule override flags
            confidence: Model confidence (0.0-1.0), lower = less certain
            maturity: Data maturity level for this domain
        """
        override_reasons: list[str] = []

        # === Override hierarchy: hard rules supersede ML score ===
        if overrides.sanctioned_country:
            override_reasons.append("SANCTIONED_COUNTRY")
            return Decision.BLOCK, RiskLevel.CRITICAL, override_reasons

        if overrides.blacklisted_device:
            override_reasons.append("BLACKLISTED_DEVICE")
            return Decision.BLOCK, RiskLevel.CRITICAL, override_reasons

        if overrides.blacklisted_ip:
            override_reasons.append("BLACKLISTED_IP")
            return Decision.BLOCK, RiskLevel.CRITICAL, override_reasons

        if overrides.blacklisted_user:
            override_reasons.append("BLACKLISTED_USER")
            return Decision.BLOCK, RiskLevel.CRITICAL, override_reasons

        # Velocity anomaly → force review regardless of score
        if overrides.velocity_anomaly:
            override_reasons.append("VELOCITY_ANOMALY")
            if score < self.thresholds.review_above:
                return Decision.REVIEW, RiskLevel.MEDIUM, override_reasons

        # New account with high amount → force review
        if overrides.new_account_high_amount:
            override_reasons.append("NEW_ACCOUNT_HIGH_AMOUNT")
            if score < self.thresholds.review_above:
                return Decision.REVIEW, RiskLevel.MEDIUM, override_reasons

        # === Confidence-aware threshold adjustment ===
        effective_thresholds = self._adjust_thresholds_for_confidence(confidence, maturity)

        # === ML score-based decision ===
        if score >= effective_thresholds["block"]:
            return Decision.BLOCK, RiskLevel.CRITICAL, override_reasons

        if score >= effective_thresholds["review"]:
            risk_level = RiskLevel.HIGH if score >= 0.80 else RiskLevel.MEDIUM
            return Decision.REVIEW, risk_level, override_reasons

        # Low confidence + medium score → MONITOR instead of APPROVE
        if confidence < 0.5 and maturity in (MaturityLevel.COLD, MaturityLevel.WARMING):
            if score >= effective_thresholds["approve"] * 0.7:
                override_reasons.append("LOW_CONFIDENCE_MONITOR")
                return Decision.MONITOR, RiskLevel.LOW, override_reasons

        if score >= effective_thresholds["approve"]:
            return Decision.APPROVE, RiskLevel.LOW, override_reasons

        return Decision.APPROVE, RiskLevel.LOW, override_reasons

    def _adjust_thresholds_for_confidence(self, confidence: float, maturity: MaturityLevel) -> dict[str, float]:
        """Adjust thresholds based on confidence and maturity.

        Low confidence + COLD/WARMING → stricter (lower) thresholds to catch more.
        High confidence + HOT → standard thresholds.
        """
        if confidence >= 0.8 and maturity == MaturityLevel.HOT:
            return {
                "approve": self.thresholds.approve_below,
                "review": self.thresholds.review_above,
                "block": self.thresholds.block_above,
            }

        # Stricter thresholds when confidence is low
        strictness = 1.0 - confidence * 0.5  # 1.0 at conf=0, 0.5 at conf=1.0
        if maturity in (MaturityLevel.COLD, MaturityLevel.WARMING):
            strictness = min(strictness * 1.3, 1.0)

        return {
            "approve": self.thresholds.approve_below * (1 - strictness * 0.2),
            "review": self.thresholds.review_above * (1 - strictness * 0.15),
            "block": self.thresholds.block_above * (1 - strictness * 0.1),
        }


@dataclass
class AMLThresholdConfig:
    """AML-specific thresholds with additional tiers."""

    medium_above: float
    high_above: float
    critical_above: float
    version: str = "atv1.0"

    # Structuring rule params
    structuring_window_hours: int = 24
    structuring_txn_count: int = 5
    velocity_multiplier: float = 3.0
    country_risk_weights: dict[str, float] | None = None

    def validate(self) -> bool:
        return self.medium_above < self.high_above < self.critical_above


class AMLDecisionEngine:
    """AML decision engine — ML scores + structuring/sanctions/velocity rules.

    Supports confidence-aware decisions and maturity-based mode escalation.
    """

    def __init__(self, thresholds: AMLThresholdConfig):
        self.thresholds = thresholds
        if not thresholds.validate():
            raise ValueError("Invalid AML thresholds")

    def evaluate(
        self,
        score: float,
        sanctions_match: bool = False,
        blacklisted_user: bool = False,
        blacklisted_device: bool = False,
        watchlist_hit: bool = False,
        structuring_detected: bool = False,
        velocity_spike: bool = False,
        confidence: float = 1.0,
        maturity: MaturityLevel = MaturityLevel.HOT,
    ) -> tuple[Decision, RiskLevel, list[str]]:
        """Evaluate AML risk and return (decision, risk_level, rule_overrides)."""
        overrides: list[str] = []

        # === Hard rule overrides (bypass ML) ===
        if sanctions_match:
            overrides.append("SANCTIONS_MATCH")
            return Decision.BLOCK, RiskLevel.CRITICAL, overrides

        if watchlist_hit:
            overrides.append("WATCHLIST_HIT")
            return Decision.BLOCK, RiskLevel.CRITICAL, overrides

        if blacklisted_user:
            overrides.append("BLACKLISTED_USER")
            return Decision.BLOCK, RiskLevel.CRITICAL, overrides

        if blacklisted_device:
            overrides.append("BLACKLISTED_DEVICE")
            return Decision.BLOCK, RiskLevel.CRITICAL, overrides

        # === Soft rule overrides (elevate to REVIEW) ===
        # In COLD/WARMING, weight rules 2x (lower the score threshold for override)
        rule_weight = 2.0 if maturity in (MaturityLevel.COLD, MaturityLevel.WARMING) else 1.0

        if structuring_detected:
            overrides.append("STRUCTURING_DETECTED")
            threshold = self.thresholds.high_above / rule_weight
            if score < threshold:
                return Decision.REVIEW, RiskLevel.HIGH, overrides

        if velocity_spike:
            overrides.append("VELOCITY_SPIKE")
            threshold = self.thresholds.high_above / rule_weight
            if score < threshold:
                return Decision.REVIEW, RiskLevel.MEDIUM, overrides

        # === Confidence-aware threshold adjustment ===
        t = self.thresholds
        if confidence < 0.5 and maturity in (MaturityLevel.COLD, MaturityLevel.WARMING):
            # Stricter thresholds
            critical = t.critical_above * 0.85
            high = t.high_above * 0.85
            medium = t.medium_above * 0.85
        elif confidence >= 0.8 and maturity == MaturityLevel.HOT:
            critical = t.critical_above
            high = t.high_above
            medium = t.medium_above
        else:
            # Moderate adjustment
            adj = 1.0 - (1.0 - confidence) * 0.15
            critical = t.critical_above * adj
            high = t.high_above * adj
            medium = t.medium_above * adj

        # === ML score-based decision ===
        if score >= critical:
            return Decision.BLOCK, RiskLevel.CRITICAL, overrides

        if score >= high:
            return Decision.REVIEW, RiskLevel.HIGH, overrides

        if score >= medium:
            return Decision.MONITOR, RiskLevel.MEDIUM, overrides

        # Low confidence + medium-ish score → MONITOR fallback
        if confidence < 0.5 and score >= medium * 0.7:
            overrides.append("LOW_CONFIDENCE_MONITOR")
            return Decision.MONITOR, RiskLevel.LOW, overrides

        return Decision.APPROVE, RiskLevel.LOW, overrides


@dataclass
class MerchantThresholdConfig:
    """Merchant-specific thresholds with tier assignment."""

    standard_below: float = 0.40
    enhanced_above: float = 0.40
    restricted_above: float = 0.65
    blocked_above: float = 0.85
    version: str = "mtv1.0"

    # Hard rule params
    chargeback_rate_block: float = 0.10
    fraud_rate_restrict: float = 0.05

    def validate(self) -> bool:
        return self.standard_below <= self.enhanced_above < self.restricted_above < self.blocked_above


class MerchantDecisionEngine:
    """Merchant decision engine — ML scores + chargeback/fraud rules → tier assignment.

    Supports confidence-aware decisions and maturity-based mode escalation.
    """

    def __init__(self, thresholds: MerchantThresholdConfig):
        self.thresholds = thresholds
        if not thresholds.validate():
            raise ValueError("Invalid merchant thresholds")

    def evaluate(
        self,
        score: float,
        chargeback_rate: float = 0.0,
        fraud_rate: float = 0.0,
        velocity_spike: bool = False,
        confidence: float = 1.0,
        maturity: MaturityLevel = MaturityLevel.HOT,
    ) -> tuple[Decision, RiskLevel, str, list[str]]:
        """Evaluate merchant risk. Returns (decision, risk_level, tier, rule_overrides)."""
        overrides: list[str] = []

        # === Hard rule overrides ===
        if chargeback_rate >= self.thresholds.chargeback_rate_block:
            overrides.append("EXCESSIVE_CHARGEBACKS")
            return Decision.BLOCK, RiskLevel.CRITICAL, "BLOCKED", overrides

        if fraud_rate >= self.thresholds.fraud_rate_restrict:
            overrides.append("HIGH_FRAUD_RATE")
            if score < self.thresholds.restricted_above:
                return Decision.REVIEW, RiskLevel.HIGH, "RESTRICTED", overrides

        if velocity_spike:
            overrides.append("VELOCITY_SPIKE")
            if score < self.thresholds.enhanced_above:
                return Decision.REVIEW, RiskLevel.MEDIUM, "ENHANCED", overrides

        # === Confidence-aware threshold adjustment ===
        t = self.thresholds
        if confidence < 0.5 and maturity in (MaturityLevel.COLD, MaturityLevel.WARMING):
            adj = 0.85
        elif confidence >= 0.8 and maturity == MaturityLevel.HOT:
            adj = 1.0
        else:
            adj = 1.0 - (1.0 - confidence) * 0.15

        blocked = t.blocked_above * adj
        restricted = t.restricted_above * adj
        enhanced = t.enhanced_above * adj

        # === ML score-based decision + tier ===
        if score >= blocked:
            return Decision.BLOCK, RiskLevel.CRITICAL, "BLOCKED", overrides

        if score >= restricted:
            return Decision.REVIEW, RiskLevel.HIGH, "RESTRICTED", overrides

        if score >= enhanced:
            return Decision.MONITOR, RiskLevel.MEDIUM, "ENHANCED", overrides

        # Low confidence fallback
        if confidence < 0.5 and score >= enhanced * 0.7:
            overrides.append("LOW_CONFIDENCE_MONITOR")
            return Decision.MONITOR, RiskLevel.LOW, "ENHANCED", overrides

        return Decision.APPROVE, RiskLevel.LOW, "STANDARD", overrides
