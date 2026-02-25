"""Unit tests for Resilience utilities (retry + circuit breaker)."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from serving.app.services.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    get_circuit_breaker,
    with_retry,
)


class TestRetryDecorator:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        call_count = 0

        @with_retry(max_attempts=3)
        async def good_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await good_func()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        call_count = 0

        @with_retry(max_attempts=3, backoff_base=0.01, backoff_max=0.05)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return "recovered"

        result = await flaky_func()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        @with_retry(max_attempts=2, backoff_base=0.01, backoff_max=0.05)
        async def always_fails():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_does_not_retry_on_value_error(self):
        call_count = 0

        @with_retry(max_attempts=3)
        async def validation_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad data")

        with pytest.raises(ValueError):
            await validation_error()
        assert call_count == 1


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Only 2 failures after reset, should still be closed
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_successes(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb._state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_call_succeeds_when_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        mock_fn = AsyncMock(return_value="data")
        result = await cb.call(mock_fn)
        assert result == "data"

    @pytest.mark.asyncio
    async def test_call_raises_when_open_no_fallback(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        mock_fn = AsyncMock(return_value="data")
        with pytest.raises(CircuitOpenError):
            await cb.call(mock_fn)
        # Function should not have been called
        mock_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_returns_fallback_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1, fallback=lambda: "default")
        cb.record_failure()
        mock_fn = AsyncMock(return_value="data")
        result = await cb.call(mock_fn)
        assert result == "default"
        mock_fn.assert_not_called()


class TestGetCircuitBreaker:
    def test_creates_singleton(self):
        # Clean state
        from serving.app.services.resilience import _circuit_breakers

        _circuit_breakers.clear()

        cb1 = get_circuit_breaker("redis")
        cb2 = get_circuit_breaker("redis")
        assert cb1 is cb2

    def test_different_names_create_different_breakers(self):
        from serving.app.services.resilience import _circuit_breakers

        _circuit_breakers.clear()

        cb_redis = get_circuit_breaker("redis")
        cb_db = get_circuit_breaker("db")
        assert cb_redis is not cb_db
