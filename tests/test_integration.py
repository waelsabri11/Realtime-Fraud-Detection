"""Needs real Postgres + Redis (CI services). Skipped locally unless DATABASE_URL points to Postgres."""

from __future__ import annotations

import os

import pytest
import redis
from sqlalchemy import create_engine, text

from fraudrt.db.repo import Repository
from fraudrt.features.online import RedisFeatureStore

DB = os.environ.get("DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DB.startswith("postgresql"), reason="DATABASE_URL not set"),
]


def test_postgres_roundtrip_after_migrations() -> None:
    engine = create_engine(DB)
    repo = Repository(engine)
    assert repo.ping()
    tid = 777_000_001
    tx = {"transaction_id": tid, "event_dt": 5, "amount": 12.5, "uid": "u", "payload": {"a": None}}
    sc = {
        "transaction_id": tid,
        "model_name": "m",
        "model_version": "1",
        "fraud_score": 0.3,
        "expected_loss": 3.75,
        "features": {"uid_cnt_1h": 0.0, "ProductCD": "W"},
        "latency_ms": 1.2,
    }
    repo.save_scored([tx], [sc])
    repo.save_scored([tx], [sc])  # idempotent
    repo.save_labels([{"transaction_id": tid, "is_fraud": True, "available_dt": 99}])
    with engine.connect() as c:
        row = c.execute(
            text(
                "select s.features->>'ProductCD', l.is_fraud from scores s "
                "join labels l using (transaction_id) where s.transaction_id = :t"
            ),
            {"t": tid},
        ).one()
        assert tuple(row) == ("W", True)
    with engine.begin() as c:
        for t in ("labels", "scores", "transactions"):
            c.execute(text(f"delete from {t} where transaction_id = :t"), {"t": tid})


def test_real_redis_feature_store() -> None:
    client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    store = RedisFeatureStore(client, prefix="fs-itest")
    store.flush()
    first, f0 = store.read_features("u1", 1, 1000, 10.0)
    assert first and f0["uid_cnt_7d"] == 0
    store.update("u1", 1, 1000, 1000)
    _, f1 = store.read_features("u1", 2, 1060, 20.0)
    assert f1["uid_cnt_1h"] == 1 and f1["uid_secs_since_last"] == 60
    store.flush()
