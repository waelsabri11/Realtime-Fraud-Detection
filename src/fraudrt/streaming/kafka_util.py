"""Kafka (Redpanda) helpers: topic creation and consumer-group lag."""

from __future__ import annotations

import logging
import time

from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore[attr-defined]

from fraudrt.config import Settings

log = logging.getLogger(__name__)


def ensure_topics(settings: Settings) -> None:
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    new = [
        NewTopic(t, num_partitions=settings.topic_partitions, replication_factor=1)
        for t in (settings.topic_transactions, settings.topic_labels)
        if t not in existing
    ]
    if not new:
        return
    for t, fut in admin.create_topics(new).items():
        fut.result()
        log.info("created topic %s", t)


def delete_topics(settings: Settings) -> None:
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    todo = [t for t in (settings.topic_transactions, settings.topic_labels) if t in existing]
    if todo:
        for fut in admin.delete_topics(todo).values():
            fut.result()
        time.sleep(1)


def lag_by_partition(settings: Settings) -> dict[tuple[str, int], int]:
    """Messages produced but not yet committed by the scorer group, per (topic, partition)."""
    c = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": settings.consumer_group,
            "enable.auto.commit": False,
        }
    )
    try:
        md = c.list_topics(timeout=10)
        topics = [settings.topic_transactions, settings.topic_labels]
        tps = [TopicPartition(t, p) for t in topics for p in md.topics[t].partitions]
        lags: dict[tuple[str, int], int] = {}
        for tp in c.committed(tps, timeout=10):
            lo, hi = c.get_watermark_offsets(tp, timeout=10)
            lags[(tp.topic, tp.partition)] = hi - (tp.offset if tp.offset >= 0 else lo)
        return lags
    finally:
        c.close()


def group_lag(settings: Settings) -> int:
    """Total lag over both topics."""
    return sum(lag_by_partition(settings).values())
