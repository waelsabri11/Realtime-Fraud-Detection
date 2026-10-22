"""Scoring core: features -> model -> response, independent of FastAPI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fraudrt.api.schemas import ScoreRequest
from fraudrt.features.definitions import amount_cents, make_uid, stateless_features
from fraudrt.features.online import RedisFeatureStore
from fraudrt.modeling.registry import ModelBundle


@dataclass
class ScoredTransaction:
    request: ScoreRequest
    uid: str
    features: dict[str, float | str | None]
    score: float
    duplicate: bool
    bundle: ModelBundle
    timings_ms: dict[str, float]

    @property
    def expected_loss(self) -> float:
        return self.score * self.request.transaction_amt


class Scorer:
    def __init__(self, store: RedisFeatureStore) -> None:
        self.store = store

    def score(self, req: ScoreRequest, bundle: ModelBundle) -> ScoredTransaction:
        t0 = time.perf_counter()
        row = req.to_row()
        uid = make_uid(row)
        feats: dict[str, Any] = dict(stateless_features(row))
        t1 = time.perf_counter()
        first, beh = self.store.read_features(
            uid, req.transaction_id, req.transaction_dt, req.transaction_amt
        )
        feats.update(beh)
        t2 = time.perf_counter()
        x = bundle.vocab.encode_row(feats).reshape(1, -1)
        score = float(bundle.predict(x)[0])
        t3 = time.perf_counter()
        if first:
            self.store.update(
                uid, req.transaction_id, req.transaction_dt, amount_cents(req.transaction_amt)
            )
        t4 = time.perf_counter()
        return ScoredTransaction(
            request=req,
            uid=uid,
            features=feats,
            score=score,
            duplicate=not first,
            bundle=bundle,
            timings_ms={
                "features": (t1 - t0) * 1e3,
                "redis_read": (t2 - t1) * 1e3,
                "model": (t3 - t2) * 1e3,
                "redis_write": (t4 - t3) * 1e3,
                "total": (t4 - t0) * 1e3,
            },
        )
