"""Small synthetic IEEE-CIS-shaped data for tests, CI and load tests (no Kaggle data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudrt.features.definitions import RAW_CATEGORICAL, RAW_NUMERIC

_CATS: dict[str, list[str]] = {
    "ProductCD": ["W", "C", "R", "H", "S"],
    "card4": ["visa", "mastercard", "discover", "american express"],
    "card6": ["debit", "credit"],
    "P_emaildomain": ["gmail.com", "yahoo.com", "outlook.com", "anonymous.com"],
    "R_emaildomain": ["gmail.com", "yahoo.com", "outlook.com", "anonymous.com"],
    "DeviceType": ["desktop", "mobile"],
    "id_12": ["NotFound", "Found"],
    "id_15": ["New", "Found", "Unknown"],
    "id_16": ["NotFound", "Found"],
    "id_28": ["New", "Found"],
    "id_29": ["NotFound", "Found"],
    "id_31": ["chrome 63.0", "safari generic", "ie 11.0", "samsung browser 6.2"],
    "id_35": ["T", "F"],
    "id_36": ["T", "F"],
    "id_37": ["T", "F"],
    "id_38": ["T", "F"],
}
_DAY = 86_400


def make_synthetic(
    n: int = 4000, n_cards: int = 300, days: int = 40, seed: int = 0
) -> pd.DataFrame:
    """Raw-column frame (TransactionID, isFraud, TransactionDT, ...) sorted by time.

    Cards repeat so per-identity windows are exercised; some timestamps are deliberately tied.
    """
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(_DAY, _DAY * (days + 1), size=n))
    dt[rng.integers(0, n, size=n // 20)] += 0  # keep ties plausible after sort
    card = rng.integers(0, n_cards, size=n)
    amt = np.round(rng.lognormal(4.0, 1.0, size=n), 2)
    amt[rng.random(n) < 0.05] = 100.0  # repeated identical amounts
    df = pd.DataFrame(
        {
            "TransactionID": 3_000_000 + np.arange(n),
            "TransactionDT": dt,
            "TransactionAmt": amt,
            "card1": 1000 + card,
            "card2": np.where(card % 7 == 0, np.nan, 100 + card % 50),
            "card3": 150.0,
            "card5": 100.0 + card % 5,
            "addr1": np.where(card % 11 == 0, np.nan, 200 + card % 30),
            "addr2": 87.0,
        }
    )
    for c in RAW_CATEGORICAL:
        if c in _CATS:
            vals = np.array(_CATS[c] + [None], dtype=object)
            if c in ("card4", "card6"):
                df[c] = np.array(_CATS[c], dtype=object)[card % len(_CATS[c])]
            elif c == "P_emaildomain":
                df[c] = np.array(_CATS[c], dtype=object)[card % len(_CATS[c])]
            else:
                df[c] = vals[rng.integers(0, len(vals), size=n)]
    for i in range(1, 10):
        df[f"M{i}"] = np.array(["T", "F", None], dtype=object)[rng.integers(0, 3, size=n)]
    for c in RAW_NUMERIC:
        if c not in df.columns:
            df[c] = np.where(rng.random(n) < 0.3, np.nan, rng.gamma(2.0, 20.0, size=n).round(1))
    risk = (
        0.8 * (df["TransactionAmt"] > 300)
        + 1.2 * (df["P_emaildomain"] == "anonymous.com")
        + 0.9 * (df["ProductCD"] == "C")
        + 0.5 * (df["P_emaildomain"] != df["R_emaildomain"])
    )
    p = 1 / (1 + np.exp(-(-4.2 + risk)))
    df["isFraud"] = (rng.random(n) < p).astype(int)
    return df
