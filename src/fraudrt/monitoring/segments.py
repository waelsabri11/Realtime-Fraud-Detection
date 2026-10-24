"""Error rates by segment at a fixed global alert rate (a proxy check, not a fairness audit)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudrt.modeling.metrics import Arr, top_fraction_mask

AMOUNT_EDGES = [0.0, 25.0, 50.0, 100.0, 250.0, 500.0, float("inf")]
AMOUNT_LABELS = ["<25", "25-50", "50-100", "100-250", "250-500", ">=500"]
# A segment is flagged when its rates rest on too little data to be read.
MIN_ROWS = 1_000
MIN_FRAUDS = 30


def amount_bucket(amount: pd.Series) -> pd.Series:
    return pd.cut(amount, AMOUNT_EDGES, labels=AMOUNT_LABELS, right=False).astype("object")


def segment_table(
    y: Arr, alert: Arr, segment: pd.Series, min_rows: int = MIN_ROWS, min_frauds: int = MIN_FRAUDS
) -> list[dict[str, Any]]:
    """Recall / precision / FPR of the alert mask within each value of `segment` (NaN -> 'missing')."""
    seg = segment.fillna("missing").astype(str).to_numpy()
    rows = []
    for value in sorted(set(seg)):
        m = seg == value
        yy, aa = y[m].astype(bool), alert[m]
        tp, fp = int((yy & aa).sum()), int((~yy & aa).sum())
        pos, neg = int(yy.sum()), int((~yy).sum())
        rows.append(
            {
                "segment": value,
                "n": int(m.sum()),
                "frauds": pos,
                "fraud_rate": pos / max(1, int(m.sum())),
                "alerts": int(aa.sum()),
                "alert_rate": float(aa.mean()),
                "recall": tp / pos if pos else None,
                "precision": tp / int(aa.sum()) if aa.sum() else None,
                "fpr": fp / neg if neg else None,
                "low_support": bool(m.sum() < min_rows or pos < min_frauds),
            }
        )
    return rows


def segment_report(
    df: pd.DataFrame, score: Arr, columns: list[str], alert_rate: float = 0.03
) -> dict[str, Any]:
    """`df` needs isFraud, TransactionAmt and the segment columns; alerts = global top `alert_rate`."""
    y = df["isFraud"].to_numpy()
    alert = top_fraction_mask(np.asarray(score), alert_rate)
    tp, fp = int((y.astype(bool) & alert).sum()), int((~y.astype(bool) & alert).sum())
    out: dict[str, Any] = {
        "alert_rate": alert_rate,
        "overall": {
            "n": len(y),
            "frauds": int(y.sum()),
            "alerts": int(alert.sum()),
            "recall": tp / int(y.sum()),
            "precision": tp / int(alert.sum()),
            "fpr": fp / int((y == 0).sum()),
        },
        "segments": {},
    }
    for c in columns:
        out["segments"][c] = segment_table(y, alert, df[c].reset_index(drop=True))
    buckets = segment_table(y, alert, amount_bucket(df["TransactionAmt"]).reset_index(drop=True))
    out["segments"]["amount_bucket"] = sorted(
        buckets, key=lambda r: AMOUNT_LABELS.index(r["segment"])
    )
    return out
