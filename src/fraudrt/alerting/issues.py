"""Issue drafting and the two backends: GitHub (REST API) and dry-run (files on disk).

One open issue per alert name: a firing alert whose `alert:<name>` label already has an open issue
is skipped; a resolved alert adds a comment to the open issue (it is left open for a human to
close, since resolution of the alert is not resolution of its cause).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

MARKER = "<!-- fraudrt-alert:{name} -->"


@dataclass
class Alert:
    name: str
    status: str  # firing | resolved
    severity: str
    summary: str
    description: str
    value: str
    starts_at: str
    window: str
    labels: dict[str, str] = field(default_factory=dict)
    generator_url: str = ""

    @classmethod
    def from_webhook(cls, raw: dict[str, Any]) -> Alert:
        labels, ann = raw.get("labels", {}), raw.get("annotations", {})
        return cls(
            name=labels.get("alertname", "unknown"),
            status=raw.get("status", "firing"),
            severity=labels.get("severity", "unknown"),
            summary=ann.get("summary", ""),
            description=ann.get("description", ""),
            value=ann.get("value", ""),
            starts_at=raw.get("startsAt", ""),
            window=ann.get("window", ""),
            labels=labels,
            generator_url=raw.get("generatorURL", ""),
        )


@dataclass
class Context:
    """What the receiver looked up in Prometheus when the alert arrived."""

    readings: dict[str, float | None] = field(default_factory=dict)
    top_features_now: list[tuple[str, float]] = field(default_factory=list)
    top_feature_increase: list[tuple[str, float, float]] = field(default_factory=list)
    baseline: str = ""


@dataclass
class IssueDraft:
    title: str
    body: str
    labels: list[str]
    alert_name: str


def label_for(alert_name: str) -> str:
    return f"alert:{alert_name}"


def render_issue(alert: Alert, ctx: Context, dashboard_url: str, now: str) -> IssueDraft:
    lines = [
        MARKER.format(name=alert.name),
        f"**Alert `{alert.name}`** ({alert.severity}) fired at `{alert.starts_at}`; "
        f"this issue was opened at `{now}`.",
        "",
        alert.summary,
        "",
        alert.description,
        "",
        "### Values",
        "",
        f"* value that triggered the rule: **{alert.value or 'n/a'}**",
        f"* rule window: {alert.window or 'see rule'}",
        f"* time window of interest: `{alert.starts_at}` to `{now}`",
    ]
    if ctx.readings:
        lines += ["", "### Readings when the alert arrived", "", "| metric | value |", "|---|---|"]
        for k, v in ctx.readings.items():
            lines.append(f"| {k} | {'n/a' if v is None else f'{v:.4g}'} |")
    if ctx.top_features_now:
        lines += [
            "",
            "### Features with the highest PSI now (rolling window vs training reference)",
            "",
            "| feature | PSI |",
            "|---|---|",
            *[f"| `{f}` | {v:.3f} |" for f, v in ctx.top_features_now],
            "",
            "Several features sit above 0.25 even in normal operation (see `docs/MODEL_CARD.md`), "
            "so the level alone is not the signal; see the largest increase below. "
            "`hour` and `dow` are left out: a window shorter than a week never matches their "
            "reference.",
        ]
    if ctx.top_feature_increase:
        lines += [
            "",
            f"### Largest PSI increase since {ctx.baseline}",
            "",
            "| feature | before | now |",
            "|---|---|---|",
            *[f"| `{f}` | {b:.3f} | {n:.3f} |" for f, b, n in ctx.top_feature_increase],
        ]
    lines += ["", f"[Dashboard]({dashboard_url})"]
    if alert.generator_url:
        lines.append(f"[Rule expression in Prometheus]({alert.generator_url})")
    return IssueDraft(
        title=f"[alert] {alert.name}: {alert.summary}"[:200],
        body="\n".join(lines) + "\n",
        labels=["monitoring", label_for(alert.name)],
        alert_name=alert.name,
    )


class IssueBackend(Protocol):
    def find_open(self, alert_name: str) -> str | None: ...
    def create(self, draft: IssueDraft) -> str: ...
    def comment(self, ref: str, text: str) -> None: ...


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class DryRunBackend:
    """Writes issue bodies to `directory/<alert>.md`; open issues are tracked in state.json."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._state = self.dir / "state.json"

    def _load(self) -> dict[str, str]:
        return json.loads(self._state.read_text()) if self._state.exists() else {}

    def find_open(self, alert_name: str) -> str | None:
        return self._load().get(alert_name)

    def create(self, draft: IssueDraft) -> str:
        path = self.dir / f"{_safe(draft.alert_name)}.md"
        header = f"# {draft.title}\n\nlabels: {', '.join(draft.labels)}\n\n---\n\n"
        path.write_text(header + draft.body)
        state = self._load()
        state[draft.alert_name] = str(path)
        self._state.write_text(json.dumps(state, indent=2))
        return str(path)

    def comment(self, ref: str, text: str) -> None:
        with open(ref, "a") as fh:
            fh.write(f"\n---\n**Comment:** {text}\n")
        state = {k: v for k, v in self._load().items() if v != ref}
        self._state.write_text(json.dumps(state, indent=2))


class GitHubBackend:
    def __init__(self, repo: str, token: str, client: httpx.Client | None = None) -> None:
        self.repo = repo
        self.client = client or httpx.Client(
            base_url="https://api.github.com",
            timeout=15,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def find_open(self, alert_name: str) -> str | None:
        r = self.client.get(
            f"/repos/{self.repo}/issues",
            params={"state": "open", "labels": label_for(alert_name), "per_page": 5},
        )
        r.raise_for_status()
        issues = [i for i in r.json() if "pull_request" not in i]
        return str(issues[0]["number"]) if issues else None

    def create(self, draft: IssueDraft) -> str:
        r = self.client.post(
            f"/repos/{self.repo}/issues",
            json={"title": draft.title, "body": draft.body, "labels": draft.labels},
        )
        r.raise_for_status()
        return str(r.json()["number"])

    def comment(self, ref: str, text: str) -> None:
        r = self.client.post(
            f"/repos/{self.repo}/issues/{quote(ref)}/comments", json={"body": text}
        )
        r.raise_for_status()


class PrometheusClient:
    def __init__(self, url: str, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(base_url=url, timeout=5)

    def query(self, expr: str, at: float | None = None) -> list[tuple[dict[str, str], float]]:
        params: dict[str, str] = {"query": expr}
        if at is not None:
            params["time"] = f"{at:.3f}"
        r = self.client.get("/api/v1/query", params=params)
        r.raise_for_status()
        return [
            (row["metric"], float(row["value"][1]))
            for row in r.json()["data"]["result"]
            if row["value"][1] not in ("NaN", "+Inf", "-Inf")
        ]


READINGS = {
    "score PSI (rolling window)": "fraud_score_psi",
    "alert rate (share of scored transactions)": (
        "sum(rate(fraud_alerts_total[2m])) / sum(rate(fraud_scored_transactions_total[2m]))"
    ),
    "p99 /score latency (s)": (
        "histogram_quantile(0.99, sum by (le) (rate(fraud_api_request_duration_seconds_bucket"
        '{route="/score"}[2m])))'
    ),
    "5xx ratio": (
        'sum(rate(fraud_api_requests_total{status=~"5.."}[2m])) / sum(rate(fraud_api_requests_total[2m]))'
    ),
    "consumer lag, transactions topic": 'sum(fraud_consumer_lag_messages{topic="transactions"})',
    "requests per second": "sum(rate(fraud_api_requests_total[2m]))",
}


CALENDAR_FEATURES = frozenset({"hour", "dow"})  # structurally high in a window shorter than a week


def _psi_by_feature(rows: list[tuple[dict[str, str], float]]) -> dict[str, float]:
    return {m["feature"]: v for m, v in rows if m.get("feature") not in CALENDAR_FEATURES}


def gather_context(
    prom: PrometheusClient, starts_at_unix: float | None, lookback_seconds: float = 300
) -> Context:
    ctx = Context()
    for name, expr in READINGS.items():
        try:
            rows = prom.query(expr)
            ctx.readings[name] = rows[0][1] if rows else None
        except httpx.HTTPError:
            ctx.readings[name] = None
    try:
        now = _psi_by_feature(prom.query("fraud_feature_psi"))
        ctx.top_features_now = sorted(now.items(), key=lambda kv: -kv[1])[:5]
        before: dict[str, float] = {}
        if starts_at_unix is not None:
            t = starts_at_unix - lookback_seconds
            before = _psi_by_feature(prom.query("fraud_feature_psi", at=t))
            ctx.baseline = "`" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) + "`"
        if not before:  # the metric did not exist yet (or no alert time): lowest value of the hour
            before = _psi_by_feature(prom.query("min_over_time(fraud_feature_psi[1h])"))
            ctx.baseline = "the lowest value in the last hour"
        rises = sorted(
            ((f, before[f], v) for f, v in now.items() if f in before),
            key=lambda t3: t3[2] - t3[1],
            reverse=True,
        )
        ctx.top_feature_increase = [r for r in rises[:5] if r[2] - r[1] > 0.01]
    except httpx.HTTPError:
        pass
    return ctx
