"""Categorical vocabulary: string category -> integer code (unseen / missing -> NaN)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraudrt.features.definitions import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


@dataclass
class Vocab:
    categories: dict[str, list[str]] = field(default_factory=dict)
    _lookup: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._lookup = {c: {v: i for i, v in enumerate(vs)} for c, vs in self.categories.items()}

    @classmethod
    def fit(cls, df: pd.DataFrame, min_count: int = 20) -> Vocab:
        cats: dict[str, list[str]] = {}
        for c in CATEGORICAL_FEATURES:
            vc = df[c].dropna().astype(str).value_counts()
            cats[c] = sorted(vc[vc >= min_count].index.tolist())
        return cls(cats)

    def encode_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a float frame with FEATURE_COLUMNS in canonical order."""
        out = df[NUMERIC_FEATURES].astype("float64").copy()
        for c in CATEGORICAL_FEATURES:
            out[c] = df[c].map(self._lookup[c]).astype("float64")
        return out[FEATURE_COLUMNS]

    def encode_row(self, feats: Mapping[str, Any]) -> np.ndarray[Any, Any]:
        arr = np.empty(len(FEATURE_COLUMNS), dtype=np.float64)
        for i, c in enumerate(FEATURE_COLUMNS):
            v = feats[c]
            if c in self._lookup:
                arr[i] = self._lookup[c].get(v, math.nan) if v is not None else math.nan
            else:
                arr[i] = v
        return arr

    def to_json(self) -> dict[str, list[str]]:
        return self.categories
