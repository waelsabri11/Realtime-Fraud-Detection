"""Windowed per-identity features, strictly from *prior* transactions (no look-ahead).

Two implementations of the same definition:
  * `behavioral_from_events`  - scalar reference, used online on the state read from Redis;
  * `behavioral_offline`      - vectorised over a whole frame, used for training.
`tests/test_parity.py` asserts they agree.

Definition, for a transaction at time t with amount a, identity u, over prior events of u
(ordered by (TransactionDT, TransactionID)), window W includes events with ts >= t - W:
  uid_cnt_{1h,24h,7d}   number of prior events in window
  uid_sum_24h           sum of their amounts
  uid_mean_7d           mean amount over the 7d window (NaN if none)
  uid_amt_ratio_7d      a / uid_mean_7d (NaN if none or mean == 0)
  uid_secs_since_last   t - ts of the previous event of u, any age (NaN if first seen)
Amounts are carried as integer cents so sums are exact in both implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from fraudrt.features.definitions import BEHAVIORAL, NAN, WINDOWS_SECONDS, make_uid
from fraudrt.util import records

W1H, W24H, W7D = (WINDOWS_SECONDS[k] for k in ("1h", "24h", "7d"))


def behavioral_from_events(
    now: int,
    amount: float,
    last_ts: int | None,
    window_events: Sequence[tuple[int, int]],
) -> dict[str, float]:
    """`window_events`: prior (ts, amount_cents) with ts in [now - 7d, now]."""
    c1 = c24 = c7 = 0
    s24 = s7 = 0
    for ts, cents in window_events:
        c7 += 1
        s7 += cents
        if ts >= now - W24H:
            c24 += 1
            s24 += cents
            if ts >= now - W1H:
                c1 += 1
    mean7 = s7 / c7 / 100.0 if c7 else NAN
    ratio = amount / mean7 if c7 and mean7 > 0 else NAN
    return {
        "uid_cnt_1h": float(c1),
        "uid_cnt_24h": float(c24),
        "uid_cnt_7d": float(c7),
        "uid_sum_24h": s24 / 100.0,
        "uid_mean_7d": mean7,
        "uid_amt_ratio_7d": ratio,
        "uid_secs_since_last": float(now - last_ts) if last_ts is not None else NAN,
    }


def add_uid(df: pd.DataFrame) -> pd.Series[str]:
    """Pseudo-identity for every row, using the same scalar function as the online path."""
    recs = records(
        df[["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "P_emaildomain"]]
    )
    return pd.Series([make_uid(r) for r in recs], index=df.index, name="uid", dtype=object)


def behavioral_offline(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised behavioural features. `df` needs TransactionID, TransactionDT, TransactionAmt, uid.

    Returns a frame indexed like `df` with the BEHAVIORAL columns.
    """
    order = np.lexsort((df["TransactionID"].to_numpy(), df["TransactionDT"].to_numpy()))
    ts_t = df["TransactionDT"].to_numpy(dtype=np.int64)[order]
    amt_t = df["TransactionAmt"].to_numpy(dtype=np.float64)[order]
    cents_t = np.rint(amt_t * 100).astype(np.int64)
    gid_t = pd.factorize(df["uid"].to_numpy()[order])[0].astype(np.int64)

    # stable sort by group keeps (ts, id) order inside each group
    gorder = np.argsort(gid_t, kind="stable")
    ts = ts_t[gorder]
    amt = amt_t[gorder]
    cents = cents_t[gorder]
    gid = gid_t[gorder]
    n = len(ts)

    idx = np.arange(n)
    is_start = np.r_[True, gid[1:] != gid[:-1]]
    group_start = np.maximum.accumulate(np.where(is_start, idx, 0))

    big = np.int64(1) << 36
    key = gid * big + ts
    csum = np.concatenate([[0], np.cumsum(cents)])

    def window(w: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        lo = np.searchsorted(key, key - w, side="left")
        lo = np.maximum(lo, group_start)
        return idx - lo, csum[idx] - csum[lo]

    c1, _ = window(W1H)
    c24, s24 = window(W24H)
    c7, s7 = window(W7D)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean7 = np.where(c7 > 0, s7 / np.maximum(c7, 1) / 100.0, np.nan)
        ratio = np.where((c7 > 0) & (mean7 > 0), amt / mean7, np.nan)
    prev_ts = np.r_[0, ts[:-1]]
    since = np.where(is_start, np.nan, ts - prev_ts)

    res = np.column_stack([c1, c24, c7, s24 / 100.0, mean7, ratio, since]).astype(np.float64)
    # undo the two permutations: group order -> time order -> original row order
    result = np.empty_like(res)
    result[order[gorder]] = res
    return pd.DataFrame(result, columns=BEHAVIORAL, index=df.index)
