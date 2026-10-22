from __future__ import annotations

import numpy as np
import pandas as pd

from fraudrt.allocation import (
    greedy_select,
    knapsack_select,
    run_allocation,
    summarise,
)
from fraudrt.config import Settings


def _brute_force(values: np.ndarray, costs: np.ndarray, cap: int) -> float:
    n = len(values)
    best = 0.0
    for mask in range(1 << n):
        c = sum(costs[i] for i in range(n) if mask >> i & 1)
        if c <= cap:
            best = max(best, sum(values[i] for i in range(n) if mask >> i & 1))
    return best


def test_knapsack_is_optimal_vs_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = 10
        values = rng.uniform(0, 100, n)
        costs = rng.integers(1, 4, n)
        cap = int(rng.integers(2, 12))
        sel = knapsack_select(values, costs, cap)
        assert costs[sel].sum() <= cap
        assert values[sel].sum() == np.float64(_brute_force(values, costs, cap)) or np.isclose(
            values[sel].sum(), _brute_force(values, costs, cap)
        )


def test_uniform_costs_knapsack_equals_greedy_by_value() -> None:
    rng = np.random.default_rng(1)
    v = rng.uniform(0, 50, 200)
    c = np.ones(200, dtype=int)
    assert np.isclose(v[knapsack_select(v, c, 30)].sum(), v[greedy_select(v, c, 30)].sum())


def test_knapsack_beats_topk_score_when_amounts_differ() -> None:
    p = np.array([0.9, 0.8, 0.3])
    amt = np.array([10.0, 20.0, 1000.0])
    cost = np.array([1, 1, 1])
    assert list(greedy_select(p, cost, 2)) == [True, True, False]
    assert list(knapsack_select(p * amt, cost, 2)) == [False, True, True]


def test_run_allocation_respects_daily_capacity() -> None:
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "event_dt": np.sort(rng.integers(0, 5 * 86_400, n)),
            "amount": rng.lognormal(4, 1.2, n),
            "fraud_score": rng.beta(0.5, 8, n),
        }
    )
    df["is_fraud"] = (rng.random(n) < df["fraud_score"]).astype(int)
    s = Settings(_env_file=None)
    alloc = run_allocation(df, s, capacity=20)
    for pol in ("topk_score", "greedy_value", "knapsack", "oracle"):
        used = alloc[alloc[pol]].groupby("day")["cost"].sum()
        assert (used <= 20).all()
    res = summarise(df, alloc)
    exp = dict(zip(res["policy"], res["expected_loss_saved"], strict=True))
    saved = dict(zip(res["policy"], res["money_saved"], strict=True))
    assert exp["knapsack"] >= exp["topk_score"]
    assert saved["oracle"] >= saved["knapsack"]
