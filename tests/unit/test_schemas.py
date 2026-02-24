"""Unit tests for Pydantic schemas."""

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from serving.app.schemas.aml import AMLFeatureVector, AMLScoreRequest
from serving.app.schemas.common import Decision, GeoLocation, RiskLevel
from serving.app.schemas.fraud import FraudFeatureVector, FraudScoreRequest


class TestFraudScoreRequest:
    def test_valid_request(self):
        req = FraudScoreRequest(
            transaction_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            amount=5000.0,
            currency="KES",
            transaction_type="WALLET_TRANSFER",
        )
        assert req.amount == 5000.0
        assert req.currency == "KES"

    def test_invalid_amount_zero(self):
        with pytest.raises(ValidationError):
            FraudScoreRequest(
                transaction_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                amount=0,
                currency="KES",
                transaction_type="WALLET_TRANSFER",
            )

    def test_invalid_currency_short(self):
        with pytest.raises(ValidationError):
            FraudScoreRequest(
                transaction_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                amount=100,
                currency="KE",
                transaction_type="WALLET_TRANSFER",
            )

    def test_geo_location_validation(self):
        geo = GeoLocation(country="KE", lat=-1.29, lng=36.82)
        assert geo.country == "KE"

    def test_geo_location_invalid_lat(self):
        with pytest.raises(ValidationError):
            GeoLocation(country="KE", lat=200, lng=36.82)


class TestFraudFeatureVector:
    def test_to_array_length(self):
        fv = FraudFeatureVector(amount=1000.0)
        arr = fv.to_array()
        assert len(arr) == 13

    def test_feature_names_length(self):
        names = FraudFeatureVector.feature_names()
        assert len(names) == 13

    def test_to_array_matches_names(self):
        fv = FraudFeatureVector(amount=500.0, transaction_velocity_1h=5)
        arr = fv.to_array()
        names = FraudFeatureVector.feature_names()
        assert len(arr) == len(names)
        # amount is index 9
        assert arr[9] == 500.0
        # velocity_1h is index 1
        assert arr[1] == 5


class TestAMLFeatureVector:
    def test_to_array_length(self):
        fv = AMLFeatureVector(amount=2000.0)
        arr = fv.to_array()
        assert len(arr) == 22

    def test_feature_names_length(self):
        names = AMLFeatureVector.feature_names()
        assert len(names) == 22


class TestEnums:
    def test_risk_levels(self):
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.CRITICAL.value == "CRITICAL"

    def test_decisions(self):
        assert Decision.APPROVE.value == "APPROVE"
        assert Decision.BLOCK.value == "BLOCK"
        assert Decision.REVIEW.value == "REVIEW"
        assert Decision.MONITOR.value == "MONITOR"
