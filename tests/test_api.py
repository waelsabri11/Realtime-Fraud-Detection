from __future__ import annotations

from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient

from fraudrt.api.app import create_app
from fraudrt.api.schemas import ScoreRequest
from fraudrt.api.scorer import ScoredTransaction
from fraudrt.config import Settings
from fraudrt.modeling.registry import ModelBundle
from fraudrt.util import records


class ListSink:
    def __init__(self) -> None:
        self.items: list[tuple[ScoredTransaction, float]] = []

    def submit(self, item: ScoredTransaction, latency_ms: float) -> None:
        self.items.append((item, latency_ms))

    def close(self) -> None:
        return None


@pytest.fixture
def sink() -> ListSink:
    return ListSink()


@pytest.fixture
def client(
    settings: Settings, bundle: ModelBundle, fake_redis: fakeredis.FakeRedis, sink: ListSink
) -> Any:
    app = create_app(settings, bundle=bundle, redis_client=fake_redis, sink=sink)
    with TestClient(app) as c:
        yield c


def _payload(raw: Any, i: int) -> dict[str, Any]:
    return ScoreRequest.from_row(records(raw)[i]).model_dump(exclude_none=True)


def test_health_ok(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == {"name": "fraud-test", "version": "1"}


def test_score_returns_probability_and_latency(client: TestClient, synthetic_raw: Any) -> None:
    r = client.post("/score", json=_payload(synthetic_raw, 0))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["latency_ms"] > 0
    assert body["expected_loss"] == pytest.approx(
        body["fraud_score"] * synthetic_raw.iloc[0].TransactionAmt
    )
    assert body["model_version"] == "1"


def test_score_updates_online_state_and_is_idempotent(
    client: TestClient, synthetic_raw: Any, sink: ListSink
) -> None:
    rows = records(synthetic_raw)
    # two transactions of the same card, then a redelivery of the first
    first = ScoreRequest.from_row(rows[0])
    second = first.model_copy(
        update={"transaction_id": 99_000_001, "transaction_dt": first.transaction_dt + 60}
    )
    r1 = client.post("/score?include_features=true", json=first.model_dump(exclude_none=True))
    r2 = client.post("/score?include_features=true", json=second.model_dump(exclude_none=True))
    assert r1.json()["features"]["uid_cnt_7d"] == 0
    assert r2.json()["features"]["uid_cnt_7d"] == 1
    assert r2.json()["features"]["uid_secs_since_last"] == 60
    r3 = client.post("/score?include_features=true", json=first.model_dump(exclude_none=True))
    assert r3.json()["duplicate"] is True
    assert r3.json()["features"]["uid_cnt_7d"] == 0  # redelivery does not see itself


def test_score_rejects_bad_payloads(client: TestClient, synthetic_raw: Any) -> None:
    good = _payload(synthetic_raw, 1)
    assert client.post("/score", json={**good, "transaction_amt": -5}).status_code == 422
    assert client.post("/score", json={**good, "unknown_field": 1}).status_code == 422
    assert client.post("/score", json={**good, "attributes": {"nope": 1}}).status_code == 422
    assert client.post("/score", json={"transaction_id": 1}).status_code == 422


def test_score_503_until_model_loaded(
    settings: Settings, fake_redis: fakeredis.FakeRedis, synthetic_raw: Any
) -> None:
    s = settings.model_copy(
        update={"mlflow_tracking_uri": "http://127.0.0.1:9", "model_reload_seconds": 60}
    )
    app = create_app(s, redis_client=fake_redis, sink=ListSink())
    with TestClient(app) as c:
        assert c.post("/score", json=_payload(synthetic_raw, 2)).status_code == 503
        assert c.get("/health").json()["status"] == "degraded"
