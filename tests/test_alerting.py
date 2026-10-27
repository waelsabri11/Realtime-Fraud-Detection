from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from fraudrt.alerting.issues import (
    Alert,
    Context,
    DryRunBackend,
    GitHubBackend,
    PrometheusClient,
    gather_context,
    render_issue,
)
from fraudrt.alerting.receiver import backend_from_env, create_receiver

ROOT = Path(__file__).resolve().parent.parent
STARTS = "2026-10-27T10:00:00.000Z"


def webhook(status: str = "firing", name: str = "ScorePSIModerate") -> dict[str, Any]:
    return {
        "status": status,
        "alerts": [
            {
                "status": status,
                "labels": {"alertname": name, "severity": "warning"},
                "annotations": {
                    "summary": "score distribution drifted (PSI above 0.1)",
                    "description": "PSI moderate band.",
                    "value": "0.142",
                    "window": "rolling window of 2000 transactions, for 30s",
                },
                "startsAt": STARTS,
                "endsAt": "2026-10-27T10:10:00.000Z",
                "generatorURL": "http://prometheus:9090/graph?g0.expr=fraud_score_psi",
            }
        ],
    }


def _prom(handler: Any) -> PrometheusClient:
    client = httpx.Client(base_url="http://prom", transport=httpx.MockTransport(handler))
    return PrometheusClient("http://prom", client)


def fake_prometheus() -> PrometheusClient:
    def handler(req: httpx.Request) -> httpx.Response:
        q = req.url.params["query"]
        earlier = "time" in req.url.params
        if q == "fraud_feature_psi":
            rows = [("TransactionAmt", 0.05 if earlier else 0.9), ("id_31", 0.63), ("D1", 0.02)]
            result = [{"metric": {"feature": f}, "value": [0, str(v)]} for f, v in rows]
        else:
            result = [{"metric": {}, "value": [0, "0.142"]}]
        return httpx.Response(200, json={"status": "success", "data": {"result": result}})

    return _prom(handler)


def test_issue_body_has_alert_values_window_features_and_dashboard() -> None:
    alert = Alert.from_webhook(webhook()["alerts"][0])
    ctx = gather_context(fake_prometheus(), 1_790_000_000.0)
    draft = render_issue(alert, ctx, "http://grafana/d/x?from=1", "2026-10-27T10:02:00.000Z")
    assert draft.labels == ["monitoring", "alert:ScorePSIModerate"]
    for needle in (
        "ScorePSIModerate",
        "0.142",
        STARTS,
        "2026-10-27T10:02:00.000Z",
        "`TransactionAmt`",
        "Largest PSI increase",
        "[Dashboard](http://grafana/d/x?from=1)",
    ):
        assert needle in draft.body
    # the feature that moved (0.05 -> 0.9) is listed as an increase; id_31 (flat) is not
    increase = draft.body.split("Largest PSI increase")[1]
    assert "TransactionAmt" in increase and "id_31" not in increase


def test_dry_run_dedup_and_resolution(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    app = create_receiver(DryRunBackend(tmp_path), fake_prometheus(), events_file=events)
    c = TestClient(app)
    first = c.post("/alerts", json=webhook()).json()["results"][0]
    assert first["action"] == "created"
    issue = tmp_path / "ScorePSIModerate.md"
    body = issue.read_text()
    assert "0.142" in body and "labels: monitoring, alert:ScorePSIModerate" in body
    # repeat notification: one open issue per alert
    assert c.post("/alerts", json=webhook()).json()["results"][0]["action"] == "deduplicated"
    assert issue.read_text() == body
    # a different alert gets its own issue
    other = c.post("/alerts", json=webhook(name="ApiErrorRateHigh")).json()["results"][0]
    assert other["action"] == "created"
    # resolved: comment, then a new firing opens a new issue
    resolved = c.post("/alerts", json=webhook("resolved")).json()["results"][0]
    assert resolved["action"] == "commented_resolved"
    assert "resolved at" in issue.read_text()
    assert c.post("/alerts", json=webhook()).json()["results"][0]["action"] == "created"
    logged = [json.loads(x) for x in events.read_text().splitlines()]
    assert [e["action"] for e in logged] == [
        "created",
        "deduplicated",
        "created",
        "commented_resolved",
        "created",
    ]


def test_receiver_survives_prometheus_being_down(tmp_path: Path) -> None:
    def down(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    c = TestClient(create_receiver(DryRunBackend(tmp_path), _prom(down)))
    assert c.post("/alerts", json=webhook()).json()["results"][0]["action"] == "created"
    assert "n/a" in (tmp_path / "ScorePSIModerate.md").read_text()


def test_github_backend_dedups_via_label() -> None:
    sent: list[httpx.Request] = []
    open_issues: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(req)
        if req.method == "GET":
            assert req.url.params["labels"] == "alert:ScorePSIModerate"
            return httpx.Response(200, json=open_issues)
        open_issues.append({"number": 7})
        return httpx.Response(201, json={"number": 7})

    client = httpx.Client(
        base_url="https://api.github.com",
        headers={"Authorization": "Bearer tkn"},
        transport=httpx.MockTransport(handler),
    )
    c = TestClient(create_receiver(GitHubBackend("owner/repo", "tkn", client), None))
    assert c.post("/alerts", json=webhook()).json()["results"][0]["ref"] == "7"
    assert c.post("/alerts", json=webhook()).json()["results"][0]["action"] == "deduplicated"
    posts = [r for r in sent if r.method == "POST"]
    assert len(posts) == 1 and posts[0].url.path == "/repos/owner/repo/issues"
    payload = json.loads(posts[0].content)
    assert payload["labels"] == ["monitoring", "alert:ScorePSIModerate"]


def test_backend_is_dry_run_by_default(tmp_path: Path) -> None:
    assert isinstance(backend_from_env({"ALERT_ISSUE_DIR": str(tmp_path)}), DryRunBackend)
    with pytest.raises(RuntimeError):
        backend_from_env({"ALERT_DRY_RUN": "false"})


def test_context_default_is_empty() -> None:
    assert Context().readings == {}


def _load(p: str) -> Any:
    return yaml.safe_load((ROOT / p).read_text())


def test_alert_rules_are_well_formed() -> None:
    rules = [r for g in _load("ops/prometheus/alerts.yml")["groups"] for r in g["rules"]]
    names = {r["alert"] for r in rules}
    assert {
        "ScoreLatencyP99High",
        "ConsumerLagGrowing",
        "ScorePSIModerate",
        "ScorePSISignificant",
        "ApiErrorRateHigh",
    } <= names
    for r in rules:
        assert r["expr"].strip() and r["for"]
        assert r["labels"]["severity"] in {"warning", "critical"}
        assert {"summary", "description", "value", "window"} <= set(r["annotations"])
    thresholds = " ".join(r["expr"] for r in rules if r["alert"].startswith("ScorePSI"))
    assert "0.1" in thresholds and "0.25" in thresholds
    inhibit = _load("ops/alertmanager/alertmanager.yml")["inhibit_rules"][0]
    assert "ScorePSISignificant" in inhibit["source_matchers"][0]
    assert "ScorePSIModerate" in inhibit["target_matchers"][0]


def test_compose_monitoring_services_are_opt_in() -> None:
    services = _load("compose.yaml")["services"]
    for name in ("prometheus", "alertmanager", "grafana", "alert-receiver"):
        assert services[name]["profiles"] == ["monitoring"]
    assert "profiles" not in services["api"] and "profiles" not in services["worker"]


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool not installed")
def test_promtool_checks_and_unit_tests() -> None:
    d = ROOT / "ops/prometheus"
    for args in (["check", "rules", "alerts.yml"], ["test", "rules", "alerts_test.yml"]):
        out = subprocess.run(["promtool", *args], cwd=d, capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr


def test_dashboard_json_is_provisionable() -> None:
    d = json.loads((ROOT / "ops/grafana/dashboards/fraud-monitoring.json").read_text())
    assert d["uid"] == "fraud-monitoring" and len(d["panels"]) >= 12
    ids = [p["id"] for p in d["panels"]]
    assert len(ids) == len(set(ids))
