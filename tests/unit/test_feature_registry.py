"""Unit tests for Feature Registry."""

from feature_engineering.feature_registry import (
    AML_FEATURE_SCHEMA,
    FRAUD_FEATURE_SCHEMA,
    FeatureRegistryManager,
    feature_registry,
)


class TestFeatureRegistry:
    def test_fraud_schema_feature_count(self):
        assert FRAUD_FEATURE_SCHEMA.feature_count == 13

    def test_aml_schema_feature_count(self):
        assert AML_FEATURE_SCHEMA.feature_count == 22

    def test_fraud_schema_hash_stable(self):
        hash1 = FRAUD_FEATURE_SCHEMA.schema_hash
        hash2 = FRAUD_FEATURE_SCHEMA.schema_hash
        assert hash1 == hash2

    def test_fraud_validates_correct_length(self):
        features = [0.0] * 13
        assert FRAUD_FEATURE_SCHEMA.validate_features(features) is True

    def test_fraud_rejects_wrong_length(self):
        features = [0.0] * 10
        assert FRAUD_FEATURE_SCHEMA.validate_features(features) is False

    def test_aml_validates_correct_length(self):
        features = [0.0] * 22
        assert AML_FEATURE_SCHEMA.validate_features(features) is True

    def test_registry_contains_both_schemas(self):
        assert feature_registry.get("fraud") is not None
        assert feature_registry.get("aml") is not None

    def test_registry_validate(self):
        assert feature_registry.validate("fraud", [0.0] * 13) is True
        assert feature_registry.validate("fraud", [0.0] * 5) is False
        assert feature_registry.validate("aml", [0.0] * 22) is True

    def test_registry_unknown_key(self):
        assert feature_registry.get("unknown") is None
        assert feature_registry.validate("unknown", []) is False
