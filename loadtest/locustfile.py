"""Locust load test for POST /score using synthetic IEEE-shaped payloads (no Kaggle data needed).

Event times start after the replay period so the load test never collides with replayed history.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from locust import FastHttpUser, task

from fraudrt.api.schemas import ScoreRequest
from fraudrt.synthetic import make_synthetic
from fraudrt.util import records

# Ids/cards are in ranges real data never uses, so `fraudrt purge-loadtest` can clean up exactly
# what a run wrote. The id base is random per run so Redis' idempotency markers from an earlier
# run do not turn this one into a stream of duplicates.
BASE_ID = 900_000_000 + random.randrange(0, 80_000) * 10_000
BASE_DT = 400 * 86_400  # well after the last replayed day (183)
CARD_OFFSET = 900_000_000
_counter = itertools.count()


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    p = ScoreRequest.from_row(row).model_dump(exclude_none=True)
    p["card1"] += CARD_OFFSET
    return p


_POOL: list[dict[str, Any]] = [
    _payload(r) for r in records(make_synthetic(n=20_000, n_cards=4_000, days=30, seed=7))
]


class ScoringUser(FastHttpUser):
    @task
    def score(self) -> None:
        n = next(_counter)
        payload = dict(random.choice(_POOL))
        payload["transaction_id"] = BASE_ID + n
        payload["transaction_dt"] = BASE_DT + n  # one simulated second per request
        with self.client.post("/score", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
