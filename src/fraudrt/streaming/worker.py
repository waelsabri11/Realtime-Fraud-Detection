"""Stream worker: consumes transactions -> POST /score; consumes labels -> Postgres."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from confluent_kafka import Consumer, KafkaError, Message

from fraudrt.config import Settings
from fraudrt.db.repo import Repository, make_engine
from fraudrt.streaming.kafka_util import ensure_topics

log = logging.getLogger(__name__)


def _score(client: httpx.Client, value: bytes) -> bool:
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
        time.sleep(delay)
        delay = min(delay * 2, 10)


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
    scored = labelled = 0
    t0 = time.perf_counter()
    last_msg = time.perf_counter()
    log.info("worker started: %s -> %s", settings.kafka_bootstrap, settings.api_url)
    try:
        while True:
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
            for m in msgs:
                err = m.error()
                if err is not None:
                    if err.code() != KafkaError._PARTITION_EOF:
                        log.error("kafka error: %s", err)
                    continue
                value = m.value()
                if value is None:
                    continue
                if m.topic() == settings.topic_transactions:
                    _score(client, value)
                    scored += 1
                else:
                    labels.append(json.loads(value))
            if labels:
                repo.save_labels(labels)
                labelled += len(labels)
            consumer.commit(asynchronous=False)
            if scored and scored // 5000 != (scored - len(msgs)) // 5000:
                log.info(
                    "scored=%d labels=%d (%.0f txn/s)",
                    scored,
                    labelled,
                    scored / (time.perf_counter() - t0),
                )
    finally:
        consumer.close()
        client.close()
