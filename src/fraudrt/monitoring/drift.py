"""Population Stability Index (PSI) for features and scores against a training-time reference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudrt.features.definitions import CATEGORICAL_FEATURES, FEATURE_COLUMNS

Arr = np.ndarray[Any, Any]
EPS = 1e-4
N_BINS = 10


def psi_from_props(expected: Arr, actual: Arr) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_edges(values: Arr, n_bins: int = N_BINS) -> list[float]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return []
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return [float(x) for x in edges]


def _bin_props(values: Arr, edges: list[float]) -> list[float]:
    """Proportions over [bins..., missing]."""
    nan = np.isnan(values)
    props = np.zeros(len(edges) + 2)
    if (~nan).any():
        idx = np.searchsorted(np.asarray(edges), values[~nan], side="right")
        props[: len(edges) + 1] = np.bincount(idx, minlength=len(edges) + 1)
    props[-1] = nan.sum()
    return [float(x) for x in props / max(1, len(values))]


def build_reference(encoded: pd.DataFrame) -> dict[str, Any]:
    """Reference distribution per model feature (numeric: decile bins + missing; categorical: codes)."""
    ref: dict[str, Any] = {}
    for c in FEATURE_COLUMNS:
        v = encoded[c].to_numpy(dtype=float)
        if c in CATEGORICAL_FEATURES:
            ref[c] = {"kind": "cat", "props": _cat_props(v)}
        else:
            edges = _numeric_edges(v)
            ref[c] = {"kind": "num", "edges": edges, "props": _bin_props(v, edges)}
    return ref


def _cat_props(values: Arr) -> dict[str, float]:
    s = pd.Series(np.where(np.isnan(values), -1, values).astype(int))
    return {str(k): float(p) for k, p in s.value_counts(normalize=True).items()}


def feature_psi(ref: dict[str, Any], values: Arr) -> float:
    if ref["kind"] == "cat":
        actual = _cat_props(values)
        keys = sorted(set(ref["props"]) | set(actual))
        return psi_from_props(
            np.array([ref["props"].get(k, 0.0) for k in keys]),
            np.array([actual.get(k, 0.0) for k in keys]),
        )
    return psi_from_props(np.array(ref["props"]), np.array(_bin_props(values, ref["edges"])))


def build_score_reference(scores: Arr) -> dict[str, Any]:
    edges = _numeric_edges(scores)
    return {"kind": "num", "edges": edges, "props": _bin_props(scores, edges)}


def psi_table(ref: dict[str, Any], encoded: pd.DataFrame) -> pd.DataFrame:
    rows = [(c, feature_psi(ref[c], encoded[c].to_numpy(dtype=float))) for c in FEATURE_COLUMNS]
    return pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)


def severity(psi: float) -> str:
    """Conventional PSI reading: <0.1 stable, 0.1-0.25 moderate, >0.25 significant shift."""
    return "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "significant"
