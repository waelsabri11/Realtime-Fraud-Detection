"""Unsupervised anomaly score (Isolation Forest) as a shared feature definition.

Fitted without labels on the numeric model features. `score_frame` (offline, batch) and
`score_row` (online, one transaction) use the same imputation and the same fitted forest, so
the two paths agree; `tests/test_parity.py` checks it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fraudrt.features.definitions import NUMERIC_FEATURES

FEATURE_NAME = "anomaly_score"


class AnomalyScorer:
    def __init__(self, seed: int = 0, n_estimators: int = 200) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.median: np.ndarray[Any, Any] | None = None
        self.model: IsolationForest | None = None

    def fit(self, df: pd.DataFrame, max_rows: int = 200_000) -> AnomalyScorer:
        sample = df.sample(min(len(df), max_rows), random_state=self.seed)
        med = sample[NUMERIC_FEATURES].median().fillna(0.0)
        self.median = med.to_numpy(dtype=np.float64)
        self.model = IsolationForest(
            n_estimators=self.n_estimators, max_samples=4096, random_state=self.seed, n_jobs=8
        ).fit(self._impute(sample[NUMERIC_FEATURES].to_numpy(dtype=np.float64)))
        return self

    def _impute(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self.median is not None
        return np.where(np.isnan(x), self.median, x)

    def score_matrix(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self.model is not None
        return np.asarray(-self.model.score_samples(self._impute(x)))

    def score_frame(self, df: pd.DataFrame) -> np.ndarray[Any, Any]:
        return self.score_matrix(df[NUMERIC_FEATURES].to_numpy(dtype=np.float64))

    def score_row(self, feats: Mapping[str, Any]) -> float:
        x = np.array([[feats[c] for c in NUMERIC_FEATURES]], dtype=np.float64)
        return float(self.score_matrix(x)[0])

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> AnomalyScorer:
        obj = joblib.load(path)
        assert isinstance(obj, AnomalyScorer)
        return obj
