"""Daily alert allocation under a fixed analyst-review capacity.

Each day the system has `capacity` review units. Reviewing case i costs c_i units (1 for an
ordinary case, `high_amount_cost` for a high-amount case that needs a senior analyst) and saves
v_i = p_i * amount_i in expectation. Choosing the cases is a 0/1 knapsack:

    maximise sum v_i x_i   s.t.   sum c_i x_i <= capacity,  x_i in {0,1}

Compared against:
  topk_score    - rank by fraud probability, take cases while they fit (the usual baseline)
  greedy_value  - rank by expected loss saved v_i (optimal when every cost is 1)
  oracle        - knapsack on the true realised value (amount of actual frauds): an upper bound
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudrt.config import SECONDS_PER_DAY, Settings

Arr = np.ndarray[Any, Any]
POLICIES = ("topk_score", "greedy_value", "knapsack")


def review_costs(amount: Arr, settings: Settings) -> Arr:
    return np.where(amount >= settings.high_amount_threshold, settings.high_amount_cost, 1)


def knapsack_select(values: Arr, costs: Arr, capacity: int, max_candidates: int = 2000) -> Arr:
    """Exact 0/1 knapsack by DP over integer capacity. Returns a boolean selection mask."""
    n = len(values)
    selected = np.zeros(n, dtype=bool)
    cand = np.argsort(-values, kind="stable")[:max_candidates]
    cand = cand[values[cand] > 0]
    best = np.zeros(capacity + 1)
    keep = np.zeros((len(cand), capacity + 1), dtype=bool)
    for j, i in enumerate(cand):
        c, v = int(costs[i]), float(values[i])
        if c > capacity:
            continue
        trial = best[: capacity + 1 - c] + v
        better = trial > best[c:]
        keep[j, c:] = better
        best[c:] = np.where(better, trial, best[c:])
    w = capacity
    for j in range(len(cand) - 1, -1, -1):
        if keep[j, w]:
            i = cand[j]
            selected[i] = True
            w -= int(costs[i])
    return selected


def greedy_select(order_key: Arr, costs: Arr, capacity: int) -> Arr:
    """Walk cases by descending key, take each one that still fits."""
    selected = np.zeros(len(order_key), dtype=bool)
    left = capacity
    for i in np.argsort(-order_key, kind="stable"):
        if left <= 0:
            break
        if costs[i] <= left:
            selected[i] = True
            left -= int(costs[i])
    return selected


def allocate_day(p: Arr, amount: Arr, costs: Arr, capacity: int, policy: str) -> Arr:
    value = p * amount
    if policy == "topk_score":
        return greedy_select(p, costs, capacity)
    if policy == "greedy_value":
        return greedy_select(value, costs, capacity)
    if policy == "knapsack":
        return knapsack_select(value, costs, capacity)
    raise ValueError(policy)


def run_allocation(df: pd.DataFrame, settings: Settings, capacity: int) -> pd.DataFrame:
    """df: transaction_id, event_dt, amount, fraud_score, is_fraud. One boolean column per policy."""
    out = df[["transaction_id"]].copy()
    out["day"] = df["event_dt"] // SECONDS_PER_DAY
    cost = review_costs(df["amount"].to_numpy(), settings)
    out["cost"] = cost
    p_all = df["fraud_score"].to_numpy()
    a_all = df["amount"].to_numpy()
    y_all = df["is_fraud"].to_numpy().astype(float)
    sel = {pol: np.zeros(len(df), dtype=bool) for pol in (*POLICIES, "oracle")}
    for idx in out.groupby("day").indices.values():
        for pol in POLICIES:
            sel[pol][idx] = allocate_day(p_all[idx], a_all[idx], cost[idx], capacity, pol)
        sel["oracle"][idx] = knapsack_select(y_all[idx] * a_all[idx], cost[idx], capacity)
    for pol, mask in sel.items():
        out[pol] = mask
    return out


def summarise(df: pd.DataFrame, alloc: pd.DataFrame) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy().astype(bool)
    amt = df["amount"].to_numpy()
    p = df["fraud_score"].to_numpy()
    total_fraud = int(y.sum())
    total_fraud_amt = float(amt[y].sum())
    rows = []
    for pol in (*POLICIES, "oracle"):
        m = alloc[pol].to_numpy()
        rows.append(
            {
                "policy": pol,
                "reviews": int(m.sum()),
                "units": int(alloc["cost"].to_numpy()[m].sum()),
                "frauds_caught": int(y[m].sum()),
                "recall": float(y[m].sum() / max(1, total_fraud)),
                "precision": float(y[m].sum() / max(1, m.sum())),
                "money_saved": float(amt[m & y].sum()),
                "money_saved_share": float(amt[m & y].sum() / max(1.0, total_fraud_amt)),
                "expected_loss_saved": float((p * amt)[m].sum()),
            }
        )
    return pd.DataFrame(rows)


def decisions_rows(df: pd.DataFrame, alloc: pd.DataFrame, policy: str) -> list[dict[str, Any]]:
    m = alloc[policy].to_numpy()
    ev = (df["fraud_score"] * df["amount"]).to_numpy()
    return [
        {
            "transaction_id": int(t),
            "policy": policy,
            "decision_day": int(d),
            "review_cost": int(c),
            "expected_loss_saved": float(e),
        }
        for t, d, c, e in zip(
            alloc["transaction_id"].to_numpy()[m],
            alloc["day"].to_numpy()[m],
            alloc["cost"].to_numpy()[m],
            ev[m],
            strict=True,
        )
    ]
