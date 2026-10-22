"""Online feature store on Redis: per-identity recent history.

Keys (prefix `fs`):
  fs:z:{uid}      sorted set, score = event ts, member = "{txn_id}:{amount_cents}" (last 7d only)
  fs:last:{uid}   ts of the most recent event of the identity (any age)
  fs:seen:{txn}   idempotency marker so a redelivered message is not counted twice
"""

from __future__ import annotations

from typing import Any

import redis

from fraudrt.features.behavioral import behavioral_from_events
from fraudrt.features.definitions import MAX_WINDOW_SECONDS


class RedisFeatureStore:
    def __init__(self, client: redis.Redis, prefix: str = "fs") -> None:
        self.r = client
        self.prefix = prefix

    def _z(self, uid: str) -> str:
        return f"{self.prefix}:z:{uid}"

    def _last(self, uid: str) -> str:
        return f"{self.prefix}:last:{uid}"

    def get_features(self, uid: str, now: int, amount: float) -> dict[str, float]:
        pipe = self.r.pipeline(transaction=False)
        pipe.get(self._last(uid))
        pipe.zrangebyscore(self._z(uid), now - MAX_WINDOW_SECONDS, now, withscores=True)
        last_raw, members = pipe.execute()
        events: list[tuple[int, int]] = []
        for member, score in members:
            m = member.decode() if isinstance(member, bytes) else member
            events.append((int(score), int(m.rsplit(":", 1)[1])))
        last_ts = int(last_raw) if last_raw is not None else None
        return behavioral_from_events(now, amount, last_ts, events)

    def mark_seen(self, txn_id: int, ttl_seconds: int = 86_400) -> bool:
        """True the first time a transaction id is seen."""
        return bool(self.r.set(f"{self.prefix}:seen:{txn_id}", 1, nx=True, ex=ttl_seconds))

    def update(self, uid: str, txn_id: int, ts: int, cents: int) -> None:
        pipe = self.r.pipeline(transaction=False)
        pipe.zadd(self._z(uid), {f"{txn_id}:{cents}": ts})
        pipe.zremrangebyscore(self._z(uid), "-inf", f"({ts - MAX_WINDOW_SECONDS}")
        pipe.set(self._last(uid), ts)
        pipe.execute()

    def bulk_load(
        self, history: dict[str, list[tuple[int, int, int]]], last_ts: dict[str, int]
    ) -> None:
        """Warm the store: history[uid] = [(ts, txn_id, cents)] already restricted to the window."""
        pipe = self.r.pipeline(transaction=False)
        for uid, evs in history.items():
            if evs:
                pipe.zadd(self._z(uid), {f"{t}:{c}": ts for ts, t, c in evs})
        for uid, ts in last_ts.items():
            pipe.set(self._last(uid), ts)
        pipe.execute()

    def flush(self) -> int:
        n = 0
        batch: list[Any] = []
        for key in self.r.scan_iter(match=f"{self.prefix}:*", count=5000):
            batch.append(key)
            if len(batch) >= 5000:
                n += self.r.delete(*batch)
                batch = []
        if batch:
            n += self.r.delete(*batch)
        return n
