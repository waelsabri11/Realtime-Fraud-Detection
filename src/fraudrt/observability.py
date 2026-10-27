"""Prometheus metrics for the API and the stream worker, and the rolling drift monitor.

Each component owns a `CollectorRegistry` (no global state, so tests and several `create_app`
calls do not collide). Metric names are part of the contract with `ops/prometheus/alerts.yml` and
the Grafana dashboard; `tests/test_observability.py` checks that every name used there exists.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fraudrt.features.definitions import FEATURE_COLUMNS
from fraudrt.monitoring.drift import feature_psi

LATENCY_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
SCORE_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
UNMATCHED = "unmatched"  # label for 404s, so scanners cannot create unbounded series


class RollingDrift:
    """Last `window` scored transactions (score + encoded feature vector), thread-safe."""

    def __init__(self, window: int) -> None:
        self.window = window
        self._scores: deque[float] = deque(maxlen=window)
        self._rows: deque[np.ndarray[Any, Any]] = deque(maxlen=window)
        self._lock = threading.Lock()
        self.model_version: str | None = None

    def add(self, score: float, x: np.ndarray[Any, Any], model_version: str) -> None:
        with self._lock:
            if model_version != self.model_version:  # new model, new reference: start over
                self._scores.clear()
                self._rows.clear()
                self.model_version = model_version
            self._scores.append(score)
            self._rows.append(x)

    def __len__(self) -> int:
        return len(self._scores)

    def snapshot(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        with self._lock:
            scores = np.fromiter(self._scores, dtype=float, count=len(self._scores))
            rows = (
                np.vstack(list(self._rows)) if self._rows else np.empty((0, len(FEATURE_COLUMNS)))
            )
        return scores, rows

    def psi(self, reference: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
        """(score PSI, {feature: PSI}) of the window against the training reference."""
        scores, rows = self.snapshot()
        feats = {
            c: feature_psi(reference["features"][c], rows[:, i])
            for i, c in enumerate(FEATURE_COLUMNS)
        }
        return feature_psi(reference["score"], scores), feats


class ApiMetrics:
    def __init__(self, alert_threshold: float, drift_window: int, min_drift_samples: int) -> None:
        self.registry = CollectorRegistry()
        self.alert_threshold = alert_threshold
        self.min_drift_samples = min_drift_samples
        r = self.registry
        self.duration = Histogram(
            "fraud_api_request_duration_seconds",
            "HTTP request duration (handler plus serialisation, excludes network)",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.requests = Counter(
            "fraud_api_requests_total", "HTTP requests", ["route", "method", "status"], registry=r
        )
        self.scores = Histogram(
            "fraud_score", "Model score of scored transactions", buckets=SCORE_BUCKETS, registry=r
        )
        self.scored = Counter(
            "fraud_scored_transactions_total",
            "Transactions scored (duplicates excluded)",
            registry=r,
        )
        self.alerts = Counter(
            "fraud_alerts_total",
            "Scored transactions at or above the alert threshold (review candidates)",
            registry=r,
        )
        self.score_psi = Gauge(
            "fraud_score_psi",
            "PSI of the rolling score window vs the training reference",
            registry=r,
        )
        self.feature_psi = Gauge(
            "fraud_feature_psi",
            "PSI of a feature over the rolling window vs the training reference",
            ["feature"],
            registry=r,
        )
        self.window_size = Gauge(
            "fraud_drift_window_size", "Transactions in the rolling drift window", registry=r
        )
        self.model_info = Gauge("fraud_model_info", "Active model", ["name", "version"], registry=r)
        self.drift = RollingDrift(drift_window)
        self._shown_model: tuple[str, str] | None = None

    def observe_scored(
        self, score: float, x: np.ndarray[Any, Any], duplicate: bool, name: str, version: str
    ) -> None:
        if self._shown_model != (name, version):
            self.model_info.clear()
            self.model_info.labels(name, version).set(1)
            self._shown_model = (name, version)
        if duplicate:
            return
        self.scores.observe(score)
        self.scored.inc()
        if score >= self.alert_threshold:
            self.alerts.inc()
        self.drift.add(score, x, version)

    def refresh_drift(self, reference: Mapping[str, Any] | None) -> bool:
        """Recompute the PSI gauges; False if there is nothing to compare yet."""
        n = len(self.drift)
        self.window_size.set(n)
        if reference is None or n < self.min_drift_samples:
            return False
        score_psi, feats = self.drift.psi(reference)
        self.score_psi.set(score_psi)
        for name, value in feats.items():
            self.feature_psi.labels(name).set(value)
        return True


class MetricsMiddleware:
    """Pure ASGI middleware (no BaseHTTPMiddleware overhead): latency, request and error counts."""

    def __init__(self, app: ASGIApp, metrics: ApiMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return
        status = 500
        t0 = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = getattr(scope.get("route"), "path", UNMATCHED)
            self.metrics.duration.labels(route).observe(time.perf_counter() - t0)
            self.metrics.requests.labels(route, scope["method"], str(status)).inc()


class DriftRefresher:
    """Background thread that refreshes the PSI gauges every `interval` seconds."""

    def __init__(
        self,
        metrics: ApiMetrics,
        reference: Callable[[], Mapping[str, Any] | None],
        interval: float,
    ) -> None:
        self.metrics = metrics
        self.reference = reference
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="drift-refresh", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.metrics.refresh_drift(self.reference())
            except Exception:  # monitoring must never take the API down
                continue


class WorkerMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.messages = Counter(
            "fraud_consumer_messages_total",
            "Messages handled by the stream worker",
            ["topic"],
            registry=r,
        )
        self.labels_stored = Counter(
            "fraud_consumer_labels_stored_total", "Labels written to Postgres", registry=r
        )
        self.score_retries = Counter(
            "fraud_consumer_score_retries_total",
            "POST /score attempts that had to be retried (API down or not ready)",
            registry=r,
        )
        self.lag = Gauge(
            "fraud_consumer_lag_messages",
            "Messages published but not yet committed by the consumer group",
            ["topic", "partition"],
            registry=r,
        )
        self.clock_day = Gauge(
            "fraud_consumer_event_clock_day",
            "Simulated day reached by the replay clock",
            registry=r,
        )
