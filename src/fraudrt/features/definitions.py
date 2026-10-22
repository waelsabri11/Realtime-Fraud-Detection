"""The single feature definition shared by offline training and the online scoring path.

`stateless_features` is a pure function of one raw transaction. The windowed behavioural features
(per pseudo-identity) live in `behavioral.py`; they have an offline vectorised implementation and an
online Redis implementation that are tested against each other.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

RAW_NUMERIC: list[str] = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
]
RAW_CATEGORICAL: list[str] = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *[f"M{i}" for i in range(1, 10)],
    "DeviceType",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]
# Fields that identify the transaction / carry time, not model inputs themselves.
RAW_META: list[str] = ["TransactionID", "TransactionDT"]
RAW_FIELDS: list[str] = list(dict.fromkeys(RAW_META + RAW_NUMERIC + RAW_CATEGORICAL))
# Raw columns that go into the pseudo-identity.
UID_FIELDS: list[str] = [
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "P_emaildomain",
]

DERIVED_NUMERIC: list[str] = ["hour", "dow", "log_amt", "amt_cents", "email_match"]
BEHAVIORAL: list[str] = [
    "uid_cnt_1h",
    "uid_cnt_24h",
    "uid_cnt_7d",
    "uid_sum_24h",
    "uid_mean_7d",
    "uid_amt_ratio_7d",
    "uid_secs_since_last",
]
WINDOWS_SECONDS: dict[str, int] = {"1h": 3_600, "24h": 86_400, "7d": 604_800}
MAX_WINDOW_SECONDS = max(WINDOWS_SECONDS.values())

NUMERIC_FEATURES: list[str] = RAW_NUMERIC + DERIVED_NUMERIC + BEHAVIORAL
CATEGORICAL_FEATURES: list[str] = list(RAW_CATEGORICAL)
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

NAN = float("nan")


def to_num(v: Any) -> float:
    """Coerce to float; missing / unparsable -> NaN."""
    if v is None:
        return NAN
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return NAN
    return float(v)


def to_cat(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v)
    return s if s != "" else None


def _key_part(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    f = float(v)
    if math.isnan(f):
        return ""
    return str(int(f)) if f == int(f) else repr(f)


def make_uid(row: Mapping[str, Any]) -> str:
    """Pseudo-identity: card1-6 + billing address + purchaser email domain."""
    return "|".join(_key_part(row.get(f)) for f in UID_FIELDS)


def amount_cents(amount: float) -> int:
    return round(amount * 100)


def stateless_features(row: Mapping[str, Any]) -> dict[str, float | str | None]:
    """Row-level features: raw passthrough + cheap derived ones. No state, no look-ahead."""
    out: dict[str, float | str | None] = {}
    for c in RAW_NUMERIC:
        out[c] = to_num(row.get(c))
    for c in RAW_CATEGORICAL:
        out[c] = to_cat(row.get(c))
    amt = to_num(row.get("TransactionAmt"))
    dt = to_num(row.get("TransactionDT"))
    out["hour"] = float(int(dt // 3600) % 24)
    out["dow"] = float(int(dt // 86_400) % 7)
    out["log_amt"] = math.log1p(amt) if amt == amt else NAN
    out["amt_cents"] = float(amount_cents(amt) % 100) if amt == amt else NAN
    p, r = out["P_emaildomain"], out["R_emaildomain"]
    out["email_match"] = NAN if p is None or r is None else float(p == r)
    return out
