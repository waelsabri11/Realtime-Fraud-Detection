"""Warm the Redis feature store with the history that precedes the replay period."""

from __future__ import annotations

import logging

import numpy as np
import redis

from fraudrt.config import Settings
from fraudrt.data import load_features
from fraudrt.features.definitions import MAX_WINDOW_SECONDS
from fraudrt.features.online import RedisFeatureStore

log = logging.getLogger(__name__)


def bootstrap_redis(settings: Settings, client: redis.Redis | None = None) -> dict[str, int]:
    client = client or redis.Redis.from_url(settings.redis_url)
    store = RedisFeatureStore(client, settings.feature_prefix)
    store.flush()
    feats = load_features(settings)
    past = feats[feats["TransactionDT"] < settings.replay_start_dt]
    last = past.groupby("uid")["TransactionDT"].max()
    recent = past[past["TransactionDT"] >= settings.replay_start_dt - MAX_WINDOW_SECONDS]
    cents = np.rint(recent["TransactionAmt"].to_numpy() * 100).astype(np.int64)
    history: dict[str, list[tuple[int, int, int]]] = {}
    for uid, ts, tid, c in zip(
        recent["uid"], recent["TransactionDT"], recent["TransactionID"], cents, strict=True
    ):
        history.setdefault(uid, []).append((int(ts), int(tid), int(c)))
    uids = list(last.index)
    for i in range(0, len(uids), 5000):
        chunk = uids[i : i + 5000]
        store.bulk_load(
            {u: history[u] for u in chunk if u in history},
            {u: int(last[u]) for u in chunk},
        )
    summary = {"identities": len(uids), "window_events": len(recent)}
    log.info("redis warmed: %s", summary)
    return summary
