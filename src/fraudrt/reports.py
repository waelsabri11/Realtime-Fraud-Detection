"""Glue for the allocation experiment: read replay results from Postgres, run policies, save."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fraudrt.allocation import POLICIES, decisions_rows, run_allocation, summarise
from fraudrt.config import Settings
from fraudrt.db.repo import Repository, make_engine
from fraudrt.monitoring.report import load_replay_frame

SWEEP = (25, 50, 100, 200, 400)


def run_allocation_report(settings: Settings, capacity: int, out_dir: Path) -> dict[str, Any]:
    engine = make_engine(settings.database_url)
    df = load_replay_frame(engine, settings.replay_start_dt)
    df = df.dropna(subset=["is_fraud"]).assign(is_fraud=lambda d: d["is_fraud"].astype(int))
    alloc = run_allocation(df, settings, capacity)
    repo = Repository(engine)
    for pol in POLICIES:
        repo.save_decisions(decisions_rows(df, alloc, pol))
    main = summarise(df, alloc)
    sweep = []
    for cap in SWEEP:
        a = run_allocation(df, settings, cap) if cap != capacity else alloc
        s = summarise(df, a)
        s.insert(0, "capacity_units_per_day", cap)
        sweep.append(s)
    n_days = int(alloc["day"].nunique())
    result = {
        "capacity_units_per_day": capacity,
        "days": n_days,
        "transactions": len(df),
        "frauds": int(df["is_fraud"].sum()),
        "fraud_amount": float(df.loc[df["is_fraud"] == 1, "amount"].sum()),
        "high_amount_threshold": settings.high_amount_threshold,
        "high_amount_cost": settings.high_amount_cost,
        "policies": main.round(4).to_dict("records"),
        "capacity_sweep": pd.concat(sweep).round(4).to_dict("records"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "allocation.json").write_text(json.dumps(result, indent=2))
    return result
