"""Label-aware performance monitoring: only labels already known at the evaluation time count."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudrt.config import SECONDS_PER_DAY


def known_at(df: pd.DataFrame, as_of_dt: int) -> pd.DataFrame:
    """Rows scored by `as_of_dt` whose label has also arrived by `as_of_dt`."""
    return df[(df["event_dt"] <= as_of_dt) & (df["available_dt"] <= as_of_dt)]


def asof_series(
    df: pd.DataFrame, start_day: int, end_day: int, step_days: int = 7, window_days: int = 14
) -> pd.DataFrame:
    """At each monitoring date T: PR-AUC over the last `window_days` of transactions *with a label
    known by T*, next to the PR-AUC the same transactions get once every label is in."""
    rows = []
    for day in range(start_day + step_days, end_day + 1, step_days):
        t = day * SECONDS_PER_DAY
        win = df[(df["event_dt"] > t - window_days * SECONDS_PER_DAY) & (df["event_dt"] <= t)]
        known = known_at(win, t)
        row = {
            "monitor_day": day,
            "scored_in_window": len(win),
            "labels_known": len(known),
            "label_coverage": len(known) / max(1, len(win)),
            "frauds_known": int(known["is_fraud"].sum()) if len(known) else 0,
        }
        ok = len(known) > 0 and known["is_fraud"].nunique() == 2
        row["pr_auc_known_labels"] = (
            float(average_precision_score(known["is_fraud"], known["fraud_score"]))
            if ok
            else np.nan
        )
        full = win.dropna(subset=["is_fraud"])
        row["pr_auc_eventual"] = (
            float(average_precision_score(full["is_fraud"], full["fraud_score"]))
            if full["is_fraud"].nunique() == 2
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)
