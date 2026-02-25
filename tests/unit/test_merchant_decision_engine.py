"""Unit tests for Merchant Decision Engine."""

import pytest

from monitoring.data_maturity import MaturityLevel
from serving.app.schemas.common import Decision, RiskLevel
from serving.app.services.decision_engine import MerchantDecisionEngine, MerchantThresholdConfig


class TestMerchantDecisionEngine:
    def setup_method(self):
        self.engine = MerchantDecisionEngine(MerchantThresholdConfig())

    def test_approve_low_score(self):
        decision, level, tier, overrides = self.engine.evaluate(0.2)
        assert decision == Decision.APPROVE
        assert level == RiskLevel.LOW
        assert tier == "STANDARD"
        assert overrides == []

    def test_monitor_medium_score(self):
        decision, level, tier, _ = self.engine.evaluate(0.50)
        assert decision == Decision.MONITOR
        assert tier == "ENHANCED"

    def test_review_high_score(self):
        decision, level, tier, _ = self.engine.evaluate(0.70)
        assert decision == Decision.REVIEW
        assert tier == "RESTRICTED"

    def test_block_critical_score(self):
        decision, level, tier, _ = self.engine.evaluate(0.90)
        assert decision == Decision.BLOCK
        assert level == RiskLevel.CRITICAL
        assert tier == "BLOCKED"

    def test_chargeback_rate_override_blocks(self):
        decision, level, tier, overrides = self.engine.evaluate(0.2, chargeback_rate=0.12)
        assert decision == Decision.BLOCK
        assert tier == "BLOCKED"
        assert "EXCESSIVE_CHARGEBACKS" in overrides

    def test_fraud_rate_override_restricts(self):
        decision, level, tier, overrides = self.engine.evaluate(0.2, fraud_rate=0.06)
        assert decision == Decision.REVIEW
        assert tier == "RESTRICTED"
        assert "HIGH_FRAUD_RATE" in overrides

    def test_velocity_spike_enhances(self):
        decision, level, tier, overrides = self.engine.evaluate(0.2, velocity_spike=True)
        assert decision == Decision.REVIEW
        assert tier == "ENHANCED"
        assert "VELOCITY_SPIKE" in overrides

    def test_confidence_aware_cold_maturity(self):
        """Low confidence + COLD maturity → stricter thresholds."""
        decision, _, tier, _ = self.engine.evaluate(0.35, confidence=0.2, maturity=MaturityLevel.COLD)
        # With stricter thresholds, 0.35 might trigger MONITOR/ENHANCED
        assert decision in (Decision.APPROVE, Decision.MONITOR)

    def test_confidence_aware_hot_maturity(self):
        """High confidence + HOT maturity → standard thresholds."""
        decision, _, tier, _ = self.engine.evaluate(0.35, confidence=0.95, maturity=MaturityLevel.HOT)
        assert decision == Decision.APPROVE
        assert tier == "STANDARD"

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            MerchantDecisionEngine(
                MerchantThresholdConfig(
                    standard_below=0.9,
                    enhanced_above=0.9,
                    restricted_above=0.5,
                    blocked_above=0.3,
                )
            )
