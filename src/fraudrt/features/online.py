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

SEEN_TTL_SECONDS = 86_400


class RedisFeatureStore:
    def __init__(self, client: redis.Redis, prefix: str = "fs") -> None:
        self.r = client
        self.prefix = prefix

    def _z(self, uid: str) -> str:
        return f"{self.prefix}:z:{uid}"

    def _last(self, uid: str) -> str:
        return f"{self.prefix}:last:{uid}"

    def read_features(
        self, uid: str, txn_id: int, now: int, amount: float
    ) -> tuple[bool, dict[str, float]]:
        """One round trip: idempotency marker + history read. Returns (first_time_seen, features).

        The transaction's own entry is excluded, so a redelivered message does not see itself.
        """
        pipe = self.r.pipeline(transaction=False)
        pipe.set(f"{self.prefix}:seen:{txn_id}", 1, nx=True, ex=SEEN_TTL_SECONDS)
        pipe.get(self._last(uid))
        pipe.zrangebyscore(self._z(uid), now - MAX_WINDOW_SECONDS, now, withscores=True)
        first, last_raw, members = pipe.execute()
        events: list[tuple[int, int]] = []
        for member, score in members:
            m = member.decode() if isinstance(member, bytes) else member
            tid, cents = m.rsplit(":", 1)
            if int(tid) != txn_id:
                events.append((int(score), int(cents)))
        last_ts = int(last_raw) if last_raw is not None else None
        return bool(first), behavioral_from_events(now, amount, last_ts, events)

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
