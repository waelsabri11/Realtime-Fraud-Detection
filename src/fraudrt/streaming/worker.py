"""Stream worker: consumes transactions -> POST /score; consumes labels -> Postgres."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import httpx
from confluent_kafka import Consumer, KafkaError, Message, TopicPartition
from prometheus_client import start_http_server

from fraudrt.config import Settings
from fraudrt.db.repo import Repository, make_engine
from fraudrt.observability import WorkerMetrics
from fraudrt.streaming.kafka_util import ensure_topics, lag_by_partition

log = logging.getLogger(__name__)


def _score(client: httpx.Client, value: bytes, metrics: WorkerMetrics | None = None) -> bool:
    """POST until the API accepts it. 503 (model not ready) / connection errors are retried."""
    delay = 0.5
    while True:
        try:
            resp = client.post(
                "/score", content=value, headers={"content-type": "application/json"}
            )
            if resp.status_code == 200:
                return True
            if resp.status_code == 422:
                log.error("rejected by schema validation, skipping: %s", resp.text[:300])
                return False
            log.warning("score returned %s, retrying", resp.status_code)
        except httpx.TransportError as exc:
            log.warning("api unreachable (%s), retrying", type(exc).__name__)
        if metrics is not None:
            metrics.score_retries.inc()
        time.sleep(delay)
        delay = min(delay * 2, 10)


class LabelGate:
    """Releases a label only once the simulated clock has reached it.

    Transactions and labels share the key (identity), hence the partition index, so the clock is
    kept per partition: the latest event time scored from that partition. Within a partition,
    transactions are in time order, so when the clock passes a label's availability time the
    transaction it belongs to has certainly been scored. The labels topic is ordered by
    availability time within each partition: a partition is paused at its first not-yet-due label
    and resumed when its clock moves past it.
    """

    def __init__(self) -> None:
        self.clock: dict[int, int] = {}
        self.held: dict[tuple[str, int], tuple[int, int]] = {}  # (topic, part) -> (offset, due)

    def advance(self, partition: int, event_dt: int) -> None:
        self.clock[partition] = max(self.clock.get(partition, 0), event_dt)

    def due(self, partition: int, available_dt: int) -> bool:
        return available_dt <= self.clock.get(partition, 0)

    def resumable(self) -> list[tuple[str, int, int]]:
        return [(t, p, off) for (t, p), (off, due) in self.held.items() if self.due(p, due)]

    def max_day(self) -> float:
        return max(self.clock.values(), default=0) / 86_400


def start_lag_reporter(settings: Settings, metrics: WorkerMetrics) -> threading.Event:
    """Report committed-offset lag from a thread of its own, so the gauge keeps moving while the
    main loop is blocked retrying /score (exactly when lag matters). Returns the stop event."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                for (topic, part), lag in lag_by_partition(settings).items():
                    metrics.lag.labels(topic, str(part)).set(max(lag, 0))
            except Exception as exc:  # lag is observability; never stop the stream for it
                log.warning("lag refresh failed: %s", exc)
            stop.wait(settings.lag_interval_seconds)

    threading.Thread(target=loop, name="lag-reporter", daemon=True).start()
    return stop


def run_worker(settings: Settings, stop_after_idle_seconds: float | None = None) -> None:
    ensure_topics(settings)
    repo = Repository(make_engine(settings.database_url))
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": settings.consumer_group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([settings.topic_transactions, settings.topic_labels])
    client = httpx.Client(
        base_url=settings.api_url,
        timeout=settings.api_timeout_seconds,
        limits=httpx.Limits(max_keepalive_connections=4),
    )
    gate = LabelGate()
    metrics = WorkerMetrics()
    if settings.worker_metrics_port:
        start_http_server(settings.worker_metrics_port, registry=metrics.registry)
    stop_lag = start_lag_reporter(settings, metrics)
    scored = labelled = 0
    t0 = time.perf_counter()
    last_msg = time.perf_counter()
    log.info("worker started: %s -> %s", settings.kafka_bootstrap, settings.api_url)
    try:
        while True:
            for t, p, off in gate.resumable():
                tp = TopicPartition(t, p, off)
                consumer.seek(tp)
                consumer.resume([tp])
                del gate.held[(t, p)]
            msgs: list[Message] = consumer.consume(num_messages=200, timeout=1.0)
            if not msgs:
                if (
                    stop_after_idle_seconds is not None
                    and time.perf_counter() - last_msg > stop_after_idle_seconds
                ):
                    return
                continue
            last_msg = time.perf_counter()
            labels: list[dict[str, Any]] = []
            commits: dict[tuple[str, int], int] = {}
            blocked: set[tuple[str, int]] = set()
            for m in msgs:
                topic, part, offset = m.topic(), m.partition(), m.offset()
                if topic is None or part is None or offset is None:
                    continue
                key = (topic, part)
                err = m.error()
                if err is not None:
                    if err.code() != KafkaError._PARTITION_EOF:
                        log.error("kafka error: %s", err)
                    continue
                value = m.value()
                if value is None or key in blocked:
                    continue
                if topic == settings.topic_transactions:
                    payload = json.loads(value)
                    if "clock" in payload:  # end-of-replay tick: time passes without traffic
                        gate.advance(part, int(payload["clock"]))
                    else:
                        _score(client, value, metrics)
                        gate.advance(part, int(payload["transaction_dt"]))
                        scored += 1
                        metrics.messages.labels(topic).inc()
                else:
                    label = json.loads(value)
                    if not gate.due(part, label["available_dt"]):
                        blocked.add(key)
                        gate.held[key] = (offset, label["available_dt"])
                        consumer.pause([TopicPartition(topic, part)])
                        continue
                    labels.append(label)
                    metrics.messages.labels(topic).inc()
                commits[key] = offset + 1
            if labels:
                repo.save_labels(labels)
                labelled += len(labels)
                metrics.labels_stored.inc(len(labels))
            metrics.clock_day.set(gate.max_day())
            if commits:
                consumer.commit(
                    offsets=[TopicPartition(t, p, o) for (t, p), o in commits.items()],
                    asynchronous=False,
                )
            if scored and scored // 5000 != (scored - len(msgs)) // 5000:
                log.info(
                    "scored=%d labels=%d clock_day=%.1f (%.0f txn/s)",
                    scored,
                    labelled,
                    gate.max_day(),
                    scored / (time.perf_counter() - t0),
                )
    finally:
        stop_lag.set()
        consumer.close()
        client.close()
