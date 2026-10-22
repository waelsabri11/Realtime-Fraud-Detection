"""IEEE-CIS loading, time-based split and offline feature building."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraudrt.config import SECONDS_PER_DAY, Settings
from fraudrt.features.behavioral import add_uid, behavioral_offline
from fraudrt.features.definitions import RAW_FIELDS, stateless_features

log = logging.getLogger(__name__)

ID_COLUMNS = [c for c in RAW_FIELDS if c.startswith("id_") or c == "DeviceType"]
TXN_COLUMNS = ["isFraud"] + [c for c in RAW_FIELDS if c not in ID_COLUMNS and c != "TransactionID"]


def load_raw(settings: Settings) -> pd.DataFrame:
    """train_transaction + train_identity (left join), only the columns the model uses."""
    cache = settings.cache_dir / "raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = settings.ieee_data_dir
    tx = pd.read_csv(d / "train_transaction.csv", usecols=["TransactionID", *TXN_COLUMNS])
    ident = pd.read_csv(d / "train_identity.csv", usecols=["TransactionID", *ID_COLUMNS])
    df = tx.merge(ident, on="TransactionID", how="left")
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def build_features(raw: pd.DataFrame, chunk: int = 50_000) -> pd.DataFrame:
    """Offline feature frame: stateless (shared scalar fn) + behavioural (vectorised) + meta."""
    cols = [c for c in RAW_FIELDS if c in raw.columns]
    parts = []
    for start in range(0, len(raw), chunk):
        recs = raw.iloc[start : start + chunk][cols].astype(object).where(
            lambda x: x.notna(), None
        )
        parts.append(pd.DataFrame([stateless_features(r) for r in recs.to_dict("records")]))
    feats = pd.concat(parts, ignore_index=True)
    meta = raw[["TransactionID", "TransactionDT", "TransactionAmt"]].reset_index(drop=True)
    meta["uid"] = add_uid(raw).reset_index(drop=True)
    beh = behavioral_offline(meta)
    out = pd.concat([meta, feats.drop(columns=["TransactionAmt"]), beh], axis=1)
    out["TransactionAmt"] = meta["TransactionAmt"]
    if "isFraud" in raw.columns:
        out["isFraud"] = raw["isFraud"].to_numpy()
    return out


def load_features(settings: Settings) -> pd.DataFrame:
    cache = settings.cache_dir / "features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raw = load_raw(settings)
    log.info("building offline features for %d rows", len(raw))
    feats = build_features(raw)
    feats.to_parquet(cache)
    return feats


def split_masks(df: pd.DataFrame, s: Settings) -> dict[str, pd.Series]:
    """Time-based split with a label-delay purge so no training label is 'from the future'.

    train : DT <  train_end - delay            (labels known when training would have run)
    val   : train_end <= DT < replay_start - delay   (labels known when the replay starts)
    replay: DT >= replay_start                 (streamed through the system)
    """
    dt = df["TransactionDT"]
    delay = s.label_delay_seconds
    return {
        "train": dt < s.train_end_dt - delay,
        "val": (dt >= s.train_end_dt) & (dt < s.replay_start_dt - delay),
        "replay": dt >= s.replay_start_dt,
    }


def day_of(dt: pd.Series) -> pd.Series:
    return dt // SECONDS_PER_DAY


def work_path(s: Settings, name: str) -> Path:
    s.work_dir.mkdir(parents=True, exist_ok=True)
    return s.work_dir / name
