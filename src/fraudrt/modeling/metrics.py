"""Fraud-appropriate metrics: PR-AUC, recall at a fixed alert budget, money saved."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

Arr = np.ndarray[Any, Any]


def top_fraction_mask(score: Arr, fraction: float) -> Arr:
    k = max(1, round(len(score) * fraction))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind="stable")[:k]] = True
    return mask


def evaluate(
    y: Arr, score: Arr, amount: Arr, budgets: tuple[float, ...] = (0.01, 0.03)
) -> dict[str, float]:
    """PR-AUC / ROC-AUC plus recall, precision and money saved when alerting the top `budget` share."""
    out: dict[str, float] = {
        "n": float(len(y)),
        "fraud_rate": float(np.mean(y)),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }
    fraud_amount = float((amount * y).sum())
    for b in budgets:
        m = top_fraction_mask(score, b)
        tp = float(y[m].sum())
        tag = f"{round(b * 100)}pct"
        out[f"recall_at_{tag}"] = tp / max(1.0, float(y.sum()))
        out[f"precision_at_{tag}"] = tp / max(1, int(m.sum()))
        out[f"money_saved_at_{tag}"] = float((amount * y)[m].sum())
        out[f"money_saved_share_at_{tag}"] = out[f"money_saved_at_{tag}"] / max(1.0, fraud_amount)
    return out
