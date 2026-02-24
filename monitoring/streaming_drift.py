"""Streaming Drift Detection — ADWIN-based concept drift detection for real-time scoring events."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class DriftAlert:
    """Alert emitted when drift is detected."""

    domain: str
    metric: str
    window_size: int
    old_mean: float
    new_mean: float
    drift_magnitude: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "metric": self.metric,
            "window_size": self.window_size,
            "old_mean": round(self.old_mean, 6),
            "new_mean": round(self.new_mean, 6),
            "drift_magnitude": round(self.drift_magnitude, 6),
            "timestamp": self.timestamp,
        }


class ADWINDetector:
    """Adaptive Windowing (ADWIN) for streaming concept drift detection.

    ADWIN maintains a variable-length window of recent observations and detects
    change by comparing the distributions of two sub-windows. When a significant
    difference is detected, the older portion is dropped.

    This is a simplified ADWIN implementation suitable for monitoring ML scoring
    distributions in real-time.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_window_size: int = 10000,
        min_window_size: int = 50,
    ):
        """
        Args:
            delta: Confidence parameter (lower = more sensitive to drift)
            max_window_size: Maximum number of observations to keep
            min_window_size: Minimum observations before checking for drift
        """
        self.delta = delta
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._window: deque[float] = deque(maxlen=max_window_size)
        self._drift_detected = False

    def add_element(self, value: float) -> bool:
        """Add a new observation and check for drift.

        Returns:
            True if drift is detected
        """
        self._window.append(value)
        self._drift_detected = False

        if len(self._window) < self.min_window_size * 2:
            return False

        # Check for drift by splitting window at various points
        window_arr = np.array(self._window)
        n = len(window_arr)

        for split in range(self.min_window_size, n - self.min_window_size):
            left = window_arr[:split]
            right = window_arr[split:]

            # Hoeffding bound for change detection
            mean_diff = abs(np.mean(left) - np.mean(right))
            n_left, n_right = len(left), len(right)
            m = 1.0 / n_left + 1.0 / n_right
            epsilon = np.sqrt(m * 0.5 * np.log(4.0 / self.delta))

            if mean_diff >= epsilon:
                # Drift detected — drop older portion
                self._window = deque(right, maxlen=self.max_window_size)
                self._drift_detected = True
                return True

        return False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def window_size(self) -> int:
        return len(self._window)

    @property
    def current_mean(self) -> float:
        if not self._window:
            return 0.0
        return float(np.mean(self._window))

    def reset(self) -> None:
        self._window.clear()
        self._drift_detected = False


class StreamingDriftMonitor:
    """Real-time drift monitor that processes scoring events and triggers alerts.

    Monitors multiple signals per domain:
    - Score distribution drift
    - Confidence drift
    - Feature value drifts
    - Positive prediction rate drift
    """

    def __init__(
        self,
        domain: str,
        delta: float = 0.002,
        alert_callback: Any = None,
    ):
        self.domain = domain
        self._alert_callback = alert_callback

        # ADWIN detectors for different signals
        self._score_detector = ADWINDetector(delta=delta)
        self._confidence_detector = ADWINDetector(delta=delta)
        self._positive_rate_detector = ADWINDetector(delta=delta, min_window_size=100)

        # Per-feature detectors (created lazily)
        self._feature_detectors: dict[str, ADWINDetector] = {}
        self._delta = delta

        self._alerts: list[DriftAlert] = []
        self._total_events = 0

    def process_event(
        self,
        score: float,
        confidence: float | None = None,
        is_positive: bool | None = None,
        features: dict[str, float] | None = None,
    ) -> list[DriftAlert]:
        """Process a single scoring event and check for drift.

        Args:
            score: ML risk score
            confidence: Model confidence for this prediction
            is_positive: Whether the prediction was positive (above threshold)
            features: Feature name → value dict for feature-level drift

        Returns:
            List of drift alerts generated (empty if no drift)
        """
        self._total_events += 1
        alerts: list[DriftAlert] = []

        # Score drift
        if self._score_detector.add_element(score):
            alert = DriftAlert(
                domain=self.domain,
                metric="score_distribution",
                window_size=self._score_detector.window_size,
                old_mean=score,  # Simplified — full ADWIN would track both
                new_mean=self._score_detector.current_mean,
                drift_magnitude=abs(score - self._score_detector.current_mean),
            )
            alerts.append(alert)

        # Confidence drift
        if confidence is not None and self._confidence_detector.add_element(confidence):
            alert = DriftAlert(
                domain=self.domain,
                metric="confidence",
                window_size=self._confidence_detector.window_size,
                old_mean=confidence,
                new_mean=self._confidence_detector.current_mean,
                drift_magnitude=abs(confidence - self._confidence_detector.current_mean),
            )
            alerts.append(alert)

        # Positive rate drift
        if is_positive is not None:
            if self._positive_rate_detector.add_element(float(is_positive)):
                alert = DriftAlert(
                    domain=self.domain,
                    metric="positive_rate",
                    window_size=self._positive_rate_detector.window_size,
                    old_mean=float(is_positive),
                    new_mean=self._positive_rate_detector.current_mean,
                    drift_magnitude=abs(float(is_positive) - self._positive_rate_detector.current_mean),
                )
                alerts.append(alert)

        # Per-feature drift (check top features only for performance)
        if features:
            for name, value in features.items():
                if name not in self._feature_detectors:
                    self._feature_detectors[name] = ADWINDetector(delta=self._delta)

                if self._feature_detectors[name].add_element(value):
                    alert = DriftAlert(
                        domain=self.domain,
                        metric=f"feature:{name}",
                        window_size=self._feature_detectors[name].window_size,
                        old_mean=value,
                        new_mean=self._feature_detectors[name].current_mean,
                        drift_magnitude=abs(value - self._feature_detectors[name].current_mean),
                    )
                    alerts.append(alert)

        # Store and optionally fire callback
        if alerts:
            self._alerts.extend(alerts)
            for alert in alerts:
                logger.warning(
                    f"Streaming drift detected [{self.domain}]: "
                    f"{alert.metric} shifted by {alert.drift_magnitude:.4f}"
                )
                if self._alert_callback:
                    try:
                        self._alert_callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

        return alerts

    def get_status(self) -> dict:
        """Get current monitoring status."""
        return {
            "domain": self.domain,
            "total_events": self._total_events,
            "total_alerts": len(self._alerts),
            "score_window_size": self._score_detector.window_size,
            "score_mean": self._score_detector.current_mean,
            "confidence_mean": self._confidence_detector.current_mean,
            "positive_rate_mean": self._positive_rate_detector.current_mean,
            "tracked_features": len(self._feature_detectors),
            "recent_alerts": [a.to_dict() for a in self._alerts[-10:]],
        }

    def get_recent_alerts(self, n: int = 50) -> list[DriftAlert]:
        """Get the most recent drift alerts."""
        return self._alerts[-n:]

    def reset(self) -> None:
        """Reset all detectors."""
        self._score_detector.reset()
        self._confidence_detector.reset()
        self._positive_rate_detector.reset()
        self._feature_detectors.clear()
        self._alerts.clear()
        self._total_events = 0


class KafkaStreamingDriftConsumer:
    """Integrates streaming drift detection with Kafka scoring events.

    Consumes from scoring event topics and feeds events to StreamingDriftMonitor.
    """

    def __init__(
        self,
        monitors: dict[str, StreamingDriftMonitor] | None = None,
        alert_service: Any = None,
    ):
        self._monitors = monitors or {}
        self._alert_service = alert_service

    def add_monitor(self, domain: str, monitor: StreamingDriftMonitor) -> None:
        self._monitors[domain] = monitor

    async def process_scoring_event(self, event: dict) -> None:
        """Process a single scoring event from Kafka.

        Expected event format:
        {
            "domain": "fraud" | "aml" | "merchant",
            "score": 0.75,
            "confidence": 0.85,
            "decision": "REVIEW",
            "features": {"feature_name": value, ...}
        }
        """
        domain = event.get("domain", "")
        monitor = self._monitors.get(domain)
        if not monitor:
            return

        is_positive = event.get("decision") in ("BLOCK", "REVIEW")

        alerts = monitor.process_event(
            score=event.get("score", 0.0),
            confidence=event.get("confidence"),
            is_positive=is_positive,
            features=event.get("features"),
        )

        # Send alerts through alert service if configured
        if alerts and self._alert_service:
            for alert in alerts:
                try:
                    await self._alert_service.send_alert(
                        alert_type="DRIFT",
                        severity="WARNING",
                        subject=f"Streaming drift detected: {domain}/{alert.metric}",
                        message=f"Drift magnitude: {alert.drift_magnitude:.4f}",
                        channels=["slack", "kafka"],
                    )
                except Exception as e:
                    logger.error(f"Failed to send drift alert: {e}")
