"""Monitoring report on the replay stream: feature / score drift and label-aware performance."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from fraudrt.config import SECONDS_PER_DAY, Settings
from fraudrt.features.definitions import BEHAVIORAL, FEATURE_COLUMNS
from fraudrt.modeling.registry import ModelBundle
from fraudrt.monitoring.drift import feature_psi, psi_table, severity
from fraudrt.monitoring.performance import asof_series

log = logging.getLogger(__name__)

_QUERY = text(
    """
    SELECT t.transaction_id, t.event_dt, t.amount::float8 AS amount,
           s.fraud_score, s.model_version, s.latency_ms, s.features::text AS features,
           l.is_fraud, l.available_dt
    FROM transactions t
    JOIN scores s USING (transaction_id)
    LEFT JOIN labels l USING (transaction_id)
    WHERE t.event_dt >= :start
    ORDER BY t.event_dt, t.transaction_id
    """
)


def load_replay_frame(engine: Engine, start_dt: int) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(_QUERY, conn, params={"start": start_dt})
    return df


def features_frame(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Encoded feature matrix of the stored (as-served) features."""
    feats = pd.DataFrame([json.loads(s) for s in df["features"]])
    for c in FEATURE_COLUMNS:
        if c not in feats:
            feats[c] = np.nan
    return bundle.vocab.encode_frame(feats[FEATURE_COLUMNS])


def drift_report(
    df: pd.DataFrame, encoded: pd.DataFrame, reference: dict[str, Any], start_dt: int
) -> dict[str, Any]:
    ref_f, ref_s = reference["features"], reference["score"]
    overall = psi_table(ref_f, encoded)
    overall["severity"] = overall["psi"].map(severity)
    week = ((df["event_dt"] - start_dt) // (7 * SECONDS_PER_DAY)).to_numpy()
    weekly_score, weekly_feat = [], []
    for w in sorted(set(week)):
        m = week == w
        if m.sum() < 500:  # a short trailing week is too noisy for PSI
            continue
        scores = df.loc[m, "fraud_score"].to_numpy()
        weekly_score.append(
            {
                "week": int(w),
                "n": int(m.sum()),
                "score_psi": feature_psi(ref_s, scores),
                "mean_score": float(scores.mean()),
            }
        )
        weekly_feat.append(
            {
                "week": int(w),
                **{c: feature_psi(ref_f[c], encoded.loc[m, c].to_numpy()) for c in BEHAVIORAL},
            }
        )
    return {
        "score_psi_overall": feature_psi(ref_s, df["fraud_score"].to_numpy()),
        "n_features": len(overall),
        "n_significant": int((overall["psi"] >= 0.25).sum()),
        "n_moderate": int(((overall["psi"] >= 0.1) & (overall["psi"] < 0.25)).sum()),
        "top_features": overall.head(15).round(4).to_dict("records"),
        "weekly_score": weekly_score,
        "weekly_behavioral_psi": weekly_feat,
    }


def build_report(
    settings: Settings, engine: Engine, bundle: ModelBundle, out_dir: Path
) -> dict[str, Any]:
    if bundle.reference is None:
        raise RuntimeError("model has no drift reference artifact")
    df = load_replay_frame(engine, settings.replay_start_dt)
    if df.empty:
        raise RuntimeError("no replayed transactions in Postgres - run `make replay` first")
    enc = features_frame(df, bundle)
    report: dict[str, Any] = {
        "model": f"{bundle.name} v{bundle.version}",
        "n_scored": len(df),
        "drift": drift_report(df, enc, bundle.reference, settings.replay_start_dt),
    }
    asof = asof_series(
        df.dropna(subset=["is_fraud"]).assign(is_fraud=lambda d: d["is_fraud"].astype(int)),
        settings.replay_start_day,
        int(df["event_dt"].max() // SECONDS_PER_DAY),
    )
    report["asof_performance"] = asof.round(4).replace({np.nan: None}).to_dict("records")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monitoring.json").write_text(json.dumps(report, indent=2))
    return report
