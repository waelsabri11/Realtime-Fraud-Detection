"""Replay the hold-out period into Redpanda in TransactionDT order.

Real time is *simulated*: events are published in event-time order and paced by `speedup`
(1 = real time, 3600 = one simulated hour per wall second, 0 = as fast as possible).
Label events are emitted on the same clock, `label_delay` after their transaction, so the
system only ever sees a label once it would have been known.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
from confluent_kafka import Producer

from fraudrt.api.schemas import ScoreRequest
from fraudrt.config import Settings
from fraudrt.data import load_raw
from fraudrt.features.behavioral import add_uid
from fraudrt.streaming.kafka_util import ensure_topics
from fraudrt.util import records

log = logging.getLogger(__name__)


def build_events(raw: pd.DataFrame, settings: Settings) -> list[tuple[int, int, str, bytes, bytes]]:
    """(event_time, order, topic, key, value) for transactions + delayed labels, time-sorted."""
    df = raw[raw["TransactionDT"] >= settings.replay_start_dt].reset_index(drop=True)
    uids = add_uid(df)
    recs = records(df)
    events: list[tuple[int, int, str, bytes, bytes]] = []
    delay = settings.label_delay_seconds
    for r, uid in zip(recs, uids, strict=True):
        tid = int(r["TransactionID"])
        dt = int(r["TransactionDT"])
        req = ScoreRequest.from_row(r)
        events.append(
            (dt, tid, settings.topic_transactions, uid.encode(), req.model_dump_json().encode())
        )
        label = {
            "transaction_id": tid,
            "is_fraud": bool(r["isFraud"]),
            "available_dt": dt + delay,
        }
        events.append(
            (dt + delay, tid, settings.topic_labels, uid.encode(), json.dumps(label).encode())
        )
    events.sort(key=lambda e: (e[0], e[2] != settings.topic_transactions, e[1]))
    return events


def replay(settings: Settings, speedup: float = 0.0, limit: int | None = None) -> dict[str, Any]:
    raw = load_raw(settings)
    if limit is not None:
        start = raw[raw["TransactionDT"] >= settings.replay_start_dt].index[0]
        raw = raw.iloc[: start + limit]
    events = build_events(raw, settings)
    ensure_topics(settings)
    producer = Producer(
        {"bootstrap.servers": settings.kafka_bootstrap, "linger.ms": 20, "compression.type": "lz4"}
    )
    t_sim0 = events[0][0]
    wall0 = time.perf_counter()
    n_tx = n_lab = 0
    for i, (ts, _, topic, key, value) in enumerate(events):
        if speedup > 0:
            ahead = (ts - t_sim0) / speedup - (time.perf_counter() - wall0)
            if ahead > 0:
                time.sleep(ahead)
        while True:
            try:
                producer.produce(topic, value=value, key=key)
                break
            except BufferError:
                producer.poll(0.1)
        if topic == settings.topic_transactions:
            n_tx += 1
        else:
            n_lab += 1
        if i % 1000 == 0:
            producer.poll(0)
    # time keeps passing after the last transaction: tick every partition so pending labels release
    tick = json.dumps({"clock": events[-1][0]}).encode()
    for part in range(settings.topic_partitions):
        producer.produce(settings.topic_transactions, value=tick, partition=part)
    producer.flush()
    wall = time.perf_counter() - wall0
    summary = {
        "transactions": n_tx,
        "labels": n_lab,
        "simulated_days": round((events[-1][0] - t_sim0) / 86_400, 1),
        "speedup": speedup,
        "wall_seconds": round(wall, 1),
    }
    log.info("replay published: %s", summary)
    return summary
