"""Measured experiment: inject drift into the replay, time injection -> alert -> issue.

Needs the stack with the monitoring profile up (`make monitoring-up`, a trained model) and the
alert receiver in dry-run mode (default). Each run: reset state, restart API + Alertmanager (empty
rolling window, no leftover alert groups), warm Redis, replay `--rate` tx/s with a field multiplied
by `scale` from transaction `drift_after`, and poll Prometheus and the receiver's event log.

All timestamps are host wall-clock seconds. t_inject is when the replayer handed the first drifted
message to the Kafka producer. Output: docs/results/alerting.json
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
ISSUES = ROOT / "work/issues"
PROM = "http://localhost:9090"
RATE = 100.0  # transactions per wall second
DRIFT_AFTER = 7000  # transactions; the 5000-transaction window is full and stable well before
TIMEOUT = 180.0  # seconds after injection
# x2 is a sensitivity point (expected not to alert), so one run suffices. The `mix` scenarios were
# added after the amount runs, when no amount scale reached PSI 0.25 (disclosed in the README).
SCENARIOS: dict[str, dict[str, Any]] = {
    "amount_x2": {"field": "transaction_amt", "scale": 2.0, "repeats": 1},
    "amount_x3": {"field": "transaction_amt", "scale": 3.0, "repeats": 3},
    "amount_x5": {"field": "transaction_amt", "scale": 5.0, "repeats": 3},
    "email_domain_all_protonmail": {
        "field": "p_emaildomain", "value": "protonmail.com", "fraction": 1.0, "repeats": 3,
    },
    "email_domain_half_protonmail": {
        "field": "p_emaildomain", "value": "protonmail.com", "fraction": 0.5, "repeats": 3,
    },
}  # fmt: skip
ALERTS = ("ScorePSIModerate", "ScorePSISignificant")


def sh(*cmd: str, **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, **kw)


def prom_alerts() -> list[dict[str, Any]]:
    r = httpx.get(f"{PROM}/api/v1/alerts", timeout=5)
    return list(r.json()["data"]["alerts"])


def prom_instant(expr: str) -> float | None:
    r = httpx.get(f"{PROM}/api/v1/query", params={"query": expr}, timeout=5).json()
    res = r["data"]["result"]
    return float(res[0]["value"][1]) if res else None


def prom_range(expr: str, start: float, end: float) -> list[tuple[float, float]]:
    r = httpx.get(
        f"{PROM}/api/v1/query_range",
        params={"query": expr, "start": start, "end": end, "step": 1},
        timeout=10,
    ).json()
    res = r["data"]["result"]
    return [(float(t), float(v)) for t, v in res[0]["values"]] if res else []


def wait_for(cond: Any, what: str, timeout: float = 180.0) -> None:
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(what)
        time.sleep(1)


def api_healthy() -> bool:
    try:
        return bool(httpx.get("http://localhost:8000/health", timeout=3).json()["status"] == "ok")
    except httpx.HTTPError:
        return False


def reset_state() -> None:
    sh("docker", "compose", "stop", "worker")
    sh("uv", "run", "fraudrt", "reset")
    sh("docker", "compose", "start", "worker")
    sh("docker", "compose", "restart", "api", "alertmanager")
    wait_for(api_healthy, "api healthy")
    sh("uv", "run", "fraudrt", "bootstrap")
    wait_for(lambda: not prom_alerts(), "alerts to clear", 240)
    for f in ISSUES.glob("*"):
        f.unlink()
    time.sleep(10)  # one Prometheus evaluation cycle with a clean slate


def read_events() -> list[dict[str, Any]]:
    f = ISSUES / "events.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def one_run(spec: dict[str, Any]) -> dict[str, Any]:
    reset_state()
    log = ROOT / "work/alerting_replay.log"
    out = ROOT / "work/alerting_replay.json"
    limit = DRIFT_AFTER + int(TIMEOUT * RATE) + 1000
    proc = subprocess.Popen(
        [
            "uv", "run", "fraudrt", "replay", "--rate", str(RATE), "--limit", str(limit),
            "--drift-after", str(DRIFT_AFTER), "--drift-field", spec["field"],
            "--drift-scale", str(spec.get("scale", 1.0)),
            "--drift-fraction", str(spec.get("fraction", 1.0)), "--no-wait", "--out", str(out),
            *(["--drift-value", spec["value"]] if "value" in spec else []),
        ],
        cwd=ROOT, stdout=log.open("w"), stderr=subprocess.STDOUT,
    )  # fmt: skip
    t_start = time.time()
    t_inject: float | None = None
    seen: dict[str, dict[str, float]] = {a: {} for a in ALERTS}
    try:
        while True:
            now = time.time()
            if t_inject is None and "drift injected" in log.read_text():
                line = next(x for x in log.read_text().splitlines() if "drift injected" in x)
                t_inject = float(line.split("unix=")[1].split(":")[0])
            for a in prom_alerts():
                name = a["labels"]["alertname"]
                if name in seen:
                    seen[name].setdefault(a["state"], now)
                    seen[name].setdefault("activeAt", _ts(a["activeAt"]))
            created = {e["alert"] for e in read_events() if e["action"] == "created"}
            done = created >= set(ALERTS)
            if t_inject is not None and (done or now - t_inject > TIMEOUT):
                break
            if proc.poll() not in (None, 0):
                raise RuntimeError("replay failed, see work/alerting_replay.log")
            time.sleep(1)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    assert t_inject is not None
    t_end = time.time()
    psi = prom_range("fraud_score_psi", t_start, t_end)
    events = [e for e in read_events() if e["received_at_unix"] >= t_start]
    run: dict[str, Any] = {
        "drift": {k: v for k, v in spec.items() if k != "repeats"},
        "t_inject_unix": round(t_inject, 3),
        "psi_before_injection_max": round(max(v for t, v in psi if t < t_inject), 4),
        "psi_max_after_injection": round(max(v for t, v in psi if t >= t_inject), 4),
        "alerts": {},
    }
    for name, thr in zip(ALERTS, (0.1, 0.25), strict=True):
        first_psi = next((t for t, v in psi if t >= t_inject and v > thr), None)
        ev = [e for e in events if e["alert"] == name]
        created = next((e for e in ev if e["action"] == "created"), None)
        s = seen[name]
        run["alerts"][name] = {
            "seconds_to_psi_gauge_above_threshold": _d(first_psi, t_inject),
            "seconds_to_alert_pending": _d(s.get("activeAt"), t_inject),
            "seconds_to_alert_firing_observed": _d(s.get("firing"), t_inject),
            "seconds_to_webhook_received": _d(created and created["received_at_unix"], t_inject),
            "seconds_to_issue_written": _d(created and created["handled_at_unix"], t_inject),
            "issue_files": sorted(p.name for p in ISSUES.glob(f"{name}.md")),
            "events": [e["action"] for e in ev],
        }
    before = [a for a in ALERTS if seen[a].get("activeAt", t_inject + 1) < t_inject]
    run["alerts_active_before_injection"] = before
    run["issue_bodies_saved"] = {}
    for name in ALERTS:
        f = ISSUES / f"{name}.md"
        if f.exists():
            run["issue_bodies_saved"][name] = f.read_text()
    return run


def _ts(s: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _d(t: float | None, t0: float) -> float | None:
    return None if t is None else round(t - t0, 1)


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = [
        "seconds_to_psi_gauge_above_threshold",
        "seconds_to_alert_pending",
        "seconds_to_alert_firing_observed",
        "seconds_to_webhook_received",
        "seconds_to_issue_written",
    ]
    for name in ALERTS:
        out[name] = {"runs_detected": sum(r["alerts"][name][keys[-1]] is not None for r in runs)}
        for k in keys:
            vals = [r["alerts"][name][k] for r in runs if r["alerts"][name][k] is not None]
            if vals:
                out[name][k] = {
                    "median": round(statistics.median(vals), 1),
                    "min": min(vals),
                    "max": max(vals),
                }
    return out


def main() -> None:
    """`python scripts/alerting_experiment.py [scenario ...]`; results are merged into the JSON."""
    import sys

    config = {
        "rate_tx_per_s": RATE,
        "drift_after_transactions": DRIFT_AFTER,
        "timeout_s_after_injection": TIMEOUT,
        "psi_window": 5000,
        "scrape_interval_s": 5,
        "rule_evaluation_interval_s": 5,
        "rule_for_s": 30,
        "alertmanager_group_wait_s": 5,
        "drift_refresh_interval_s": 5,
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
    }
    path = ROOT / "docs/results/alerting.json"
    res: dict[str, Any] = (
        json.loads(path.read_text()) if path.exists() else {"config": config, "scenarios": {}}
    )
    for name in sys.argv[1:] or list(SCENARIOS):
        spec = SCENARIOS[name]
        runs = []
        for i in range(spec["repeats"]):
            print(f"{name} run {i + 1}/{spec['repeats']}", flush=True)
            r = one_run(spec)
            print(json.dumps({k: v for k, v in r.items() if k != "issue_bodies_saved"}), flush=True)
            runs.append(r)
        bodies = runs[0].pop("issue_bodies_saved")
        for r in runs[1:]:
            r.pop("issue_bodies_saved")
        res["scenarios"][name] = {
            "drift": runs[0]["drift"],
            "runs": runs,
            "summary": summarise(runs),
            "example_issue_bodies": bodies,
        }
        path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
