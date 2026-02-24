"""Unit tests for Fraud and AML Decision Engines."""

import pytest

from serving.app.schemas.common import Decision, RiskLevel
from serving.app.schemas.fraud import OverrideFlags
from serving.app.services.decision_engine import (
    AMLDecisionEngine,
    AMLThresholdConfig,
    FraudDecisionEngine,
    ThresholdConfig,
)


# === Fraud Decision Engine Tests ===


class TestFraudDecisionEngine:
    def setup_method(self):
        self.engine = FraudDecisionEngine(
            ThresholdConfig(approve_below=0.60, review_above=0.75, block_above=0.90)
        )

    def test_approve_low_score(self):
        decision, level, overrides = self.engine.evaluate(0.3, OverrideFlags())
        assert decision == Decision.APPROVE
        assert level == RiskLevel.LOW
        assert overrides == []

    def test_approve_medium_score(self):
        decision, level, _ = self.engine.evaluate(0.55, OverrideFlags())
        assert decision == Decision.APPROVE
        assert level == RiskLevel.LOW

    def test_review_high_score(self):
        decision, level, _ = self.engine.evaluate(0.78, OverrideFlags())
        assert decision == Decision.REVIEW

    def test_block_critical_score(self):
        decision, level, _ = self.engine.evaluate(0.95, OverrideFlags())
        assert decision == Decision.BLOCK
        assert level == RiskLevel.CRITICAL

    def test_block_exact_threshold(self):
        decision, _, _ = self.engine.evaluate(0.90, OverrideFlags())
        assert decision == Decision.BLOCK

    def test_override_blacklisted_device(self):
        decision, level, overrides = self.engine.evaluate(
            0.1, OverrideFlags(blacklisted_device=True)
        )
        assert decision == Decision.BLOCK
        assert level == RiskLevel.CRITICAL
        assert "BLACKLISTED_DEVICE" in overrides

    def test_override_sanctioned_country(self):
        decision, _, overrides = self.engine.evaluate(
            0.05, OverrideFlags(sanctioned_country=True)
        )
        assert decision == Decision.BLOCK
        assert "SANCTIONED_COUNTRY" in overrides

    def test_override_blacklisted_ip(self):
        decision, _, overrides = self.engine.evaluate(
            0.2, OverrideFlags(blacklisted_ip=True)
        )
        assert decision == Decision.BLOCK
        assert "BLACKLISTED_IP" in overrides

    def test_override_velocity_anomaly_elevates(self):
        decision, _, overrides = self.engine.evaluate(
            0.5, OverrideFlags(velocity_anomaly=True)
        )
        assert decision == Decision.REVIEW
        assert "VELOCITY_ANOMALY" in overrides

    def test_override_velocity_no_effect_on_high_score(self):
        decision, _, _ = self.engine.evaluate(
            0.85, OverrideFlags(velocity_anomaly=True)
        )
        assert decision == Decision.REVIEW  # Already in review range

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            FraudDecisionEngine(ThresholdConfig(approve_below=0.9, review_above=0.5, block_above=0.3))


# === AML Decision Engine Tests ===


class TestAMLDecisionEngine:
    def setup_method(self):
        self.engine = AMLDecisionEngine(
            AMLThresholdConfig(medium_above=0.50, high_above=0.70, critical_above=0.85)
        )

    def test_approve_low_score(self):
        decision, level, _ = self.engine.evaluate(0.3)
        assert decision == Decision.APPROVE
        assert level == RiskLevel.LOW

    def test_monitor_medium_score(self):
        decision, level, _ = self.engine.evaluate(0.55)
        assert decision == Decision.MONITOR
        assert level == RiskLevel.MEDIUM

    def test_review_high_score(self):
        decision, level, _ = self.engine.evaluate(0.75)
        assert decision == Decision.REVIEW
        assert level == RiskLevel.HIGH

    def test_block_critical_score(self):
        decision, level, _ = self.engine.evaluate(0.90)
        assert decision == Decision.BLOCK
        assert level == RiskLevel.CRITICAL

    def test_sanctions_match_blocks(self):
        decision, _, overrides = self.engine.evaluate(0.1, sanctions_match=True)
        assert decision == Decision.BLOCK
        assert "SANCTIONS_MATCH" in overrides

    def test_watchlist_hit_blocks(self):
        decision, _, overrides = self.engine.evaluate(0.1, watchlist_hit=True)
        assert decision == Decision.BLOCK
        assert "WATCHLIST_HIT" in overrides

    def test_structuring_detected_elevates(self):
        decision, _, overrides = self.engine.evaluate(0.4, structuring_detected=True)
        assert decision == Decision.REVIEW
        assert "STRUCTURING_DETECTED" in overrides

    def test_velocity_spike_elevates(self):
        decision, _, overrides = self.engine.evaluate(0.3, velocity_spike=True)
        assert decision == Decision.REVIEW
        assert "VELOCITY_SPIKE" in overrides

    def test_override_priority_sanctions_first(self):
        """Sanctions override should take precedence."""
        decision, _, overrides = self.engine.evaluate(
            0.1, sanctions_match=True, structuring_detected=True, velocity_spike=True
        )
        assert decision == Decision.BLOCK
        assert overrides[0] == "SANCTIONS_MATCH"
