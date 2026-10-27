"""Alertmanager webhook receiver: creates (or dry-runs) one GitHub issue per firing alert."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from fraudrt.alerting.issues import (
    Alert,
    Context,
    DryRunBackend,
    GitHubBackend,
    IssueBackend,
    PrometheusClient,
    gather_context,
    render_issue,
)

log = logging.getLogger("fraudrt.alerting")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_iso(s: str) -> float | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def backend_from_env(env: dict[str, str] | os._Environ[str] = os.environ) -> IssueBackend:
    """Dry-run unless ALERT_DRY_RUN=false *and* GITHUB_TOKEN and GITHUB_REPOSITORY are set."""
    out = Path(env.get("ALERT_ISSUE_DIR", "work/issues"))
    if env.get("ALERT_DRY_RUN", "true").lower() != "false":
        return DryRunBackend(out)
    token, repo = env.get("GITHUB_TOKEN"), env.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise RuntimeError("ALERT_DRY_RUN=false needs GITHUB_TOKEN and GITHUB_REPOSITORY")
    return GitHubBackend(repo, token)


def create_receiver(
    backend: IssueBackend,
    prom: PrometheusClient | None,
    dashboard_url: str = "http://localhost:3000/d/fraud-monitoring/fraud-monitoring",
    events_file: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="alert -> issue receiver")

    def log_event(event: dict[str, Any]) -> None:
        if events_file is None:
            return
        events_file.parent.mkdir(parents=True, exist_ok=True)
        with open(events_file, "a") as fh:
            fh.write(json.dumps(event) + "\n")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/alerts")
    async def alerts(request: Request) -> dict[str, Any]:
        payload = await request.json()
        received = time.time()
        results: list[dict[str, Any]] = []
        for raw in payload.get("alerts", []):
            alert = Alert.from_webhook(raw)
            try:
                ref = backend.find_open(alert.name)
                if alert.status == "resolved":
                    if ref is not None:
                        backend.comment(ref, f"Alert resolved at {raw.get('endsAt', 'n/a')}.")
                    action = "commented_resolved" if ref else "ignored_resolved"
                elif ref is not None:
                    action = "deduplicated"
                else:
                    starts = _parse_iso(alert.starts_at)
                    ctx = gather_context(prom, starts) if prom else None
                    draft = render_issue(
                        alert,
                        ctx or Context(),
                        dashboard_url + _time_range(starts, received),
                        _iso(received),
                    )
                    ref = backend.create(draft)
                    action = "created"
            except Exception as exc:
                log.exception("issue handling failed for %s", alert.name)
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            done = time.time()
            event = {
                "alert": alert.name,
                "status": alert.status,
                "action": action,
                "ref": ref,
                "alert_starts_at": alert.starts_at,
                "received_at": _iso(received),
                "received_at_unix": received,
                "handled_at": _iso(done),
                "handled_at_unix": done,
            }
            log_event(event)
            results.append(event)
            log.info("%s %s -> %s", alert.name, alert.status, action)
        return {"results": results}

    return app


def _time_range(starts: float | None, now: float) -> str:
    start = (starts or now) - 15 * 60
    return f"?from={int(start * 1000)}&to={int((now + 5 * 60) * 1000)}"


def app_factory() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    prom_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    out = Path(os.environ.get("ALERT_ISSUE_DIR", "work/issues"))
    return create_receiver(
        backend_from_env(),
        PrometheusClient(prom_url),
        os.environ.get("GRAFANA_URL", "http://localhost:3000").rstrip("/")
        + "/d/fraud-monitoring/fraud-monitoring",
        events_file=out / "events.jsonl",
    )
