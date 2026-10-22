"""Retraining simulation that only uses labels that would be known at the retraining date.

At `as_of_day` the training set is every transaction (offline history + replayed transactions as
served, features exactly as the model saw them) whose label has arrived by then. The challenger
is registered but not promoted; it is compared with the champion on the replay days after `as_of`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from fraudrt.config import SECONDS_PER_DAY, Settings
from fraudrt.data import load_features
from fraudrt.features.definitions import FEATURE_COLUMNS
from fraudrt.modeling.metrics import evaluate
from fraudrt.modeling.registry import ModelBundle
from fraudrt.modeling.train import train_from_frames
from fraudrt.monitoring.report import load_replay_frame

log = logging.getLogger(__name__)
VAL_DAYS = 14


def labelled_pool(
    settings: Settings, engine: Engine, as_of_dt: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(known-label training pool, replay frame with eventual labels and served scores)."""
    delay = settings.label_delay_seconds
    hist = load_features(settings)
    hist = hist[hist["TransactionDT"] < settings.replay_start_dt]
    hist = hist[hist["TransactionDT"] + delay <= as_of_dt]  # label arrived by as_of
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", *FEATURE_COLUMNS]
    replay = load_replay_frame(engine, settings.replay_start_dt)
    feats = pd.DataFrame([json.loads(s) for s in replay["features"]])
    feats = feats.reindex(columns=FEATURE_COLUMNS)
    served = pd.concat([replay.drop(columns=["features"]).reset_index(drop=True), feats], axis=1)
    known = served[(served["available_dt"] <= as_of_dt) & (served["event_dt"] <= as_of_dt)]
    known = known.rename(
        columns={
            "transaction_id": "TransactionID",
            "event_dt": "TransactionDT",
            "amount": "TransactionAmt",
        }
    ).assign(isFraud=lambda d: d["is_fraud"].astype(int))
    pool = pd.concat([hist[cols], known[cols]], ignore_index=True)
    return pool, served


def run_retrain(
    settings: Settings, engine: Engine, champion: ModelBundle, as_of_day: int
) -> dict[str, Any]:
    as_of = as_of_day * SECONDS_PER_DAY
    pool, served = labelled_pool(settings, engine, as_of)
    delay = settings.label_delay_seconds
    # newest slice of known labels is validation; everything older trains
    val_start = as_of - delay - VAL_DAYS * SECONDS_PER_DAY
    dt = pool["TransactionDT"]
    train_mask = dt < val_start
    val_mask = (dt >= val_start) & (dt <= as_of - delay)
    version, metrics = train_from_frames(
        settings,
        pool,
        train_mask,
        val_mask,
        None,
        promote=False,
        run_name=f"retrain-asof-day{as_of_day}",
        tags={"kind": "retrain", "as_of_day": str(as_of_day)},
    )
    # compare on the replay days after the retraining date, eventual labels, as-served features
    after = served[(served["event_dt"] > as_of) & served["is_fraud"].notna()]
    y = after["is_fraud"].astype(int).to_numpy()
    amt = after["amount"].to_numpy()
    champ = evaluate(y, after["fraud_score"].to_numpy(), amt)
    from fraudrt.modeling.registry import load_from_registry

    challenger_settings = settings.model_copy(update={"model_version": version})
    challenger = load_from_registry(challenger_settings)
    enc = challenger.vocab.encode_frame(after[FEATURE_COLUMNS])
    chall = evaluate(y, np.asarray(challenger.booster.predict(enc)), amt)
    result = {
        "as_of_day": as_of_day,
        "label_delay_days": settings.label_delay_days,
        "n_train": int(train_mask.sum()),
        "n_train_from_replay": int(
            (pool.loc[train_mask, "TransactionDT"] >= settings.replay_start_dt).sum()
        ),
        "challenger_version": version,
        "champion_version": champion.version,
        "evaluated_on_n": len(after),
        "evaluated_days": [as_of_day + 1, int(after["event_dt"].max() // SECONDS_PER_DAY)],
        "champion": champ,
        "challenger": chall,
        "train_metrics": metrics,
    }
    return result
