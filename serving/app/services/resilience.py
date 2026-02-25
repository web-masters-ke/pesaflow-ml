"""Resilience utilities — Retry decorators and circuit breaker for external service calls.

Uses tenacity for retry logic and a lightweight state-machine circuit breaker.
"""

import asyncio
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Type

from loguru import logger
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

# === Retry Decorator ===

# Exceptions that should trigger a retry (transient infrastructure failures)
RETRYABLE_EXCEPTIONS: tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# Attempt to include asyncpg connection errors if available
try:
    import asyncpg

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, asyncpg.PostgresConnectionError)
except ImportError:
    pass

# Attempt to include Redis connection errors
try:
    import redis.exceptions

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, redis.exceptions.ConnectionError, redis.exceptions.TimeoutError)
except ImportError:
    pass


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 0.1,
    backoff_max: float = 2.0,
    retryable_exceptions: tuple[Type[Exception], ...] | None = None,
) -> Callable:
    """Decorator that retries async functions on transient infrastructure failures.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        backoff_base: Base wait time in seconds for exponential backoff.
        backoff_max: Maximum wait time in seconds.
        retryable_exceptions: Tuple of exception types to retry on. Defaults to RETRYABLE_EXCEPTIONS.
    """
    retry_on = retryable_exceptions or RETRYABLE_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=backoff_base, max=backoff_max),
                    retry=retry_if_exception_type(retry_on),
                    reraise=True,
                ):
                    with attempt:
                        return await func(*args, **kwargs)
            except RetryError:
                logger.error(f"All {max_attempts} retry attempts exhausted for {func.__name__}")
                raise

        return wrapper

    return decorator


# === Circuit Breaker ===


class CircuitState(str, Enum):
    CLOSED = "CLOSED"  # Normal operation, requests pass through
    OPEN = "OPEN"  # Failing, requests short-circuit to fallback
    HALF_OPEN = "HALF_OPEN"  # Testing recovery, allow one request through


class CircuitBreaker:
    """Lightweight circuit breaker that prevents cascading failures.

    State machine: CLOSED -> OPEN (after failure_threshold failures) -> HALF_OPEN (after cooldown) -> CLOSED/OPEN

    Args:
        service_name: Identifier for this circuit (e.g. 'redis', 'db', 'sanctions').
        failure_threshold: Number of consecutive failures before opening the circuit.
        cooldown_seconds: Time in seconds before attempting recovery (OPEN -> HALF_OPEN).
        fallback: Optional callable returning a default value when circuit is open.
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        fallback: Callable | None = None,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count_in_half_open = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                logger.info(f"Circuit [{self.service_name}] transitioning to HALF_OPEN")
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count_in_half_open += 1
            if self._success_count_in_half_open >= 2:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count_in_half_open = 0
                logger.info(f"Circuit [{self.service_name}] CLOSED (recovered)")
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        self._success_count_in_half_open = 0

        if self._failure_count >= self.failure_threshold and self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit [{self.service_name}] OPENED after {self._failure_count} consecutive failures")

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count_in_half_open = 0

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function through the circuit breaker.

        If the circuit is OPEN, returns the fallback value immediately.
        If the circuit is HALF_OPEN, allows the call through to test recovery.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            logger.debug(f"Circuit [{self.service_name}] is OPEN, returning fallback")
            if self._fallback:
                return self._fallback()
            raise CircuitOpenError(f"Circuit [{self.service_name}] is OPEN")

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and no fallback is configured."""

    pass


# === Pre-configured circuit breakers for common services ===

_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    cooldown_seconds: float = 30.0,
    fallback: Callable | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker for a named service."""
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
            fallback=fallback,
        )
    return _circuit_breakers[service_name]
