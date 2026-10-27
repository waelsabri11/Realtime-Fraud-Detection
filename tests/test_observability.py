from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, cast

import fakeredis
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from fraudrt.api.app import create_app
from fraudrt.api.schemas import ScoreRequest
from fraudrt.config import Settings
from fraudrt.data import build_features
from fraudrt.features.definitions import FEATURE_COLUMNS
from fraudrt.modeling.registry import ModelBundle
from fraudrt.monitoring.drift import build_reference, build_score_reference
from fraudrt.observability import ApiMetrics, RollingDrift, WorkerMetrics
from fraudrt.streaming import worker
from fraudrt.streaming.replayer import DriftSpec, apply_drift
from fraudrt.util import records
from tests.test_api import ListSink

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reference(bundle: ModelBundle, synthetic_raw: Any) -> dict[str, Any]:
    feats = build_features(synthetic_raw)
    enc = bundle.vocab.encode_frame(feats[FEATURE_COLUMNS])
    scores = bundle.predict(enc.to_numpy())
    return {"features": build_reference(enc), "score": build_score_reference(scores)}


@pytest.fixture
def served(settings: Settings, bundle: ModelBundle, reference: dict[str, Any]) -> Any:
    b = ModelBundle(bundle.booster, bundle.vocab, bundle.name, bundle.version, reference=reference)
    app = create_app(settings, bundle=b, redis_client=fakeredis.FakeRedis(), sink=ListSink())
    with TestClient(app) as c:
        yield c


def _payload(raw: Any, i: int) -> dict[str, Any]:
    return ScoreRequest.from_row(records(raw)[i]).model_dump(exclude_none=True)


def _families(registry: CollectorRegistry) -> set[str]:
    names: set[str] = set()
    for fam in registry.collect():
        for suffix in ("", "_total", "_bucket", "_count", "_sum"):
            names.add(fam.name + suffix)
    return names


def test_metrics_endpoint_counts_requests_and_scores(
    served: TestClient, synthetic_raw: Any
) -> None:
    for i in range(5):
        assert served.post("/score", json=_payload(synthetic_raw, i)).status_code == 200
    assert served.post("/score", json={"transaction_id": 1}).status_code == 422
    assert served.get("/nope").status_code == 404
    text = served.get("/metrics").text
    assert 'fraud_api_requests_total{method="POST",route="/score",status="200"} 5.0' in text
    assert 'status="422"' in text
    assert 'route="unmatched"' in text  # unknown paths cannot create unbounded label values
    assert "fraud_scored_transactions_total 5.0" in text
    assert 'fraud_api_request_duration_seconds_bucket{le="0.005",route="/score"}' in text
    assert 'fraud_model_info{name="fraud-test",version="1"} 1.0' in text
    assert "/metrics" not in re.findall(r'route="([^"]+)"', text)


def test_duplicates_are_not_counted_as_scored(served: TestClient, synthetic_raw: Any) -> None:
    p = _payload(synthetic_raw, 0)
    served.post("/score", json=p)
    served.post("/score", json=p)
    assert "fraud_scored_transactions_total 1.0" in served.get("/metrics").text


def test_rolling_psi_low_on_reference_like_traffic_high_on_shift(
    bundle: ModelBundle, reference: dict[str, Any], synthetic_raw: Any
) -> None:
    feats = build_features(synthetic_raw)
    x = bundle.vocab.encode_frame(feats[FEATURE_COLUMNS]).to_numpy()
    scores = bundle.predict(x)
    m = ApiMetrics(0.3, drift_window=2000, min_drift_samples=500)
    assert m.refresh_drift(reference) is False  # nothing in the window yet
    for s, row in zip(scores[:2000], x[:2000], strict=True):
        m.observe_scored(float(s), row, False, "fraud-test", "1")
    assert m.refresh_drift(reference)
    assert m.score_psi._value.get() < 0.1
    for row in x[:2000]:  # the whole window now scores 0.95
        m.observe_scored(0.95, row, False, "fraud-test", "1")
    m.refresh_drift(reference)
    assert m.score_psi._value.get() > 0.25


def test_window_resets_when_model_version_changes() -> None:
    w = RollingDrift(10)
    w.add(0.1, np.zeros(len(FEATURE_COLUMNS)), "1")
    w.add(0.2, np.zeros(len(FEATURE_COLUMNS)), "2")
    assert len(w) == 1


def test_alert_counter_uses_threshold() -> None:
    m = ApiMetrics(0.5, 100, 10)
    x = np.zeros(len(FEATURE_COLUMNS))
    for s in (0.1, 0.5, 0.9):
        m.observe_scored(s, x, False, "n", "1")
    assert m.alerts._value.get() == 2


def test_worker_retry_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    calls = iter([503, 200])
    client = httpx.Client(
        base_url="http://api", transport=httpx.MockTransport(lambda r: httpx.Response(next(calls)))
    )
    wm = WorkerMetrics()
    assert worker._score(client, b"{}", wm)
    assert wm.score_retries._value.get() == 1


def test_apply_drift_scales_only_the_chosen_field() -> None:
    out = json.loads(apply_drift(b'{"transaction_amt": 10.0, "card1": 5}', DriftSpec(0, scale=3)))
    assert out == {"transaction_amt": 30.0, "card1": 5}


def _metric_names(text: str) -> set[str]:
    return set(re.findall(r"\bfraud_[a-z0-9_]+", text))


def test_rules_and_dashboard_only_use_metrics_that_exist(served: TestClient) -> None:
    api_registry = cast(Any, served.app).state.metrics.registry
    exported = _families(api_registry) | _families(WorkerMetrics().registry)
    used = _metric_names((ROOT / "ops/prometheus/alerts.yml").read_text())
    used |= _metric_names((ROOT / "ops/grafana/dashboards/fraud-monitoring.json").read_text())
    used |= _metric_names((ROOT / "src/fraudrt/alerting/issues.py").read_text())
    assert used, "no metric names found"
    assert used <= exported, f"unknown metrics: {sorted(used - exported)}"
