"""Semi-supervised learning under delayed labels (offline experiment, tracked in MLflow).

At retrain time T (= start of the replay period) and label delay d, only transactions with
TransactionDT <= T - d have a known label; the d most recent days are *unlabelled*. Methods:

  supervised   LightGBM on the labelled part (baseline)
  self_neg     + pseudo-labelled confident negatives from the unlabelled window
  self_negpos  + confident negatives and a capped number of confident positives
  pu           Elkan-Noto PU learning: known frauds vs everything else (unlabelled)
  anomaly      supervised + Isolation Forest score (fitted on all rows before T, no labels)

All are evaluated on the same time-based hold-out (DT >= T) with the v1 metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from fraudrt.allocation import run_allocation, summarise
from fraudrt.config import SECONDS_PER_DAY, Settings
from fraudrt.features.anomaly import AnomalyScorer
from fraudrt.features.encoding import Vocab
from fraudrt.modeling.metrics import evaluate
from fraudrt.modeling.train import default_params, fit_booster

log = logging.getLogger(__name__)
VAL_DAYS = 21
METHODS = ("supervised", "self_neg", "self_negpos", "pu", "anomaly")
NEG_THRESHOLD = 0.01  # calibrated p(fraud) below this -> pseudo-negative
POS_THRESHOLD = 0.70  # calibrated p(fraud) above this -> pseudo-positive candidate
POS_CAP_SHARE = 0.5  # at most this share of the *expected* frauds in the unlabelled window


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    unlabelled: pd.DataFrame
    holdout: pd.DataFrame


def make_split(feats: pd.DataFrame, s: Settings, delay_days: float) -> Split:
    t = s.replay_start_dt
    known_until = t - int(delay_days * SECONDS_PER_DAY)
    dt = feats["TransactionDT"]
    val_start = known_until - VAL_DAYS * SECONDS_PER_DAY
    return Split(
        train=feats[dt < val_start],
        val=feats[(dt >= val_start) & (dt < known_until)],
        unlabelled=feats[(dt >= known_until) & (dt < t)],
        holdout=feats[dt >= t],
    )


def _fit(train: pd.DataFrame, val: pd.DataFrame, seed: int, label: str = "isFraud") -> Any:
    vocab = Vocab.fit(train)
    params = default_params(seed)
    t = train.assign(isFraud=train[label]) if label != "isFraud" else train
    v = val.assign(isFraud=val[label]) if label != "isFraud" else val
    return vocab, fit_booster(t, v, vocab, params)


def _predict(vocab: Vocab, booster: Any, df: pd.DataFrame) -> np.ndarray[Any, Any]:
    return np.asarray(booster.predict(vocab.encode_frame(df), num_iteration=booster.best_iteration))


def _pseudo_labels(
    vocab: Vocab,
    booster: Any,
    split: Split,
    use_positives: bool,
    prior: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pv = _predict(vocab, booster, split.val)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        pv, split.val["isFraud"].to_numpy()
    )
    u = split.unlabelled
    p = iso.predict(_predict(vocab, booster, u))
    neg = p < NEG_THRESHOLD
    pos = np.zeros(len(u), dtype=bool)
    if use_positives:
        cand = np.where(p >= POS_THRESHOLD)[0]
        cap = int(POS_CAP_SHARE * prior * len(u))
        pos[cand[np.argsort(-p[cand])[:cap]]] = True
    chosen = neg | pos
    out = u[chosen].assign(isFraud=pos[chosen].astype(int))
    y_true = u["isFraud"].to_numpy()
    diag = {
        "pseudo_neg": float(neg.sum()),
        "pseudo_pos": float(pos.sum()),
        "pseudo_neg_true_fraud_rate": float(y_true[neg].mean()) if neg.any() else 0.0,
        "pseudo_pos_precision": float(y_true[pos].mean()) if pos.any() else 0.0,
    }
    return out, diag


def run_method(
    method: str, split: Split, seed: int, anomaly: AnomalyScorer | None
) -> tuple[np.ndarray[Any, Any], dict[str, float]]:
    """Returns hold-out scores (calibrated to p(fraud) for PU) and method diagnostics."""
    prior = float(split.train["isFraud"].mean())
    if method == "supervised":
        vocab, b = _fit(split.train, split.val, seed)
        return _predict(vocab, b, split.holdout), {"best_iteration": float(b.best_iteration)}
    if method in ("self_neg", "self_negpos"):
        vocab, b = _fit(split.train, split.val, seed)
        if len(split.unlabelled) == 0:
            return _predict(vocab, b, split.holdout), {}
        pseudo, diag = _pseudo_labels(vocab, b, split, method == "self_negpos", prior)
        vocab2, b2 = _fit(pd.concat([split.train, pseudo]), split.val, seed)
        diag["best_iteration"] = float(b2.best_iteration)
        return _predict(vocab2, b2, split.holdout), diag
    if method == "pu":
        # positives = known frauds; everything else (labelled negatives + unlabelled window) = unlabelled
        train = pd.concat(
            [split.train.assign(s=split.train["isFraud"]), split.unlabelled.assign(s=0)]
        )
        val = split.val.assign(s=lambda d: d["isFraud"])
        vocab, b = _fit(train, val, seed, label="s")
        c = float(_predict(vocab, b, val[val["s"] == 1]).mean())
        g = _predict(vocab, b, split.holdout)
        return np.clip(g / max(c, 1e-6), 0, 1), {"elkan_noto_c": c}
    if method == "anomaly":
        assert anomaly is not None
        add = lambda d: d.assign(anomaly_score=anomaly.score_frame(d))  # noqa: E731
        return _fit_with_anomaly(split, seed, add)
    raise ValueError(method)


def _fit_with_anomaly(
    split: Split, seed: int, add: Any
) -> tuple[np.ndarray[Any, Any], dict[str, float]]:
    import lightgbm as lgb

    from fraudrt.features.definitions import CATEGORICAL_FEATURES

    tr, va, ho = add(split.train), add(split.val), add(split.holdout)
    vocab = Vocab.fit(tr)

    def enc(d: pd.DataFrame) -> pd.DataFrame:
        x = vocab.encode_frame(d)
        x["anomaly_score"] = d["anomaly_score"].to_numpy()
        return x

    cat = list(CATEGORICAL_FEATURES)
    dtr = lgb.Dataset(enc(tr), tr["isFraud"], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(enc(va), va["isFraud"], categorical_feature=cat, reference=dtr)
    b = lgb.train(
        default_params(seed),
        dtr,
        3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return np.asarray(b.predict(enc(ho), num_iteration=b.best_iteration)), {
        "best_iteration": float(b.best_iteration)
    }


def holdout_metrics(
    hold: pd.DataFrame, score: np.ndarray[Any, Any], settings: Settings
) -> dict[str, float]:
    y = hold["isFraud"].to_numpy()
    amt = hold["TransactionAmt"].to_numpy()
    m = evaluate(y, score, amt)
    first_seen = (hold["uid_cnt_7d"] == 0).to_numpy() & hold[
        "uid_secs_since_last"
    ].isna().to_numpy()
    m["cold_start_n"] = float(first_seen.sum())
    m["cold_start_pr_auc"] = (
        float(average_precision_score(y[first_seen], score[first_seen]))
        if y[first_seen].sum() > 0
        else float("nan")
    )
    df = pd.DataFrame(
        {
            "transaction_id": hold["TransactionID"].to_numpy(),
            "event_dt": hold["TransactionDT"].to_numpy(),
            "amount": amt,
            "fraud_score": score,
            "is_fraud": y,
        }
    )
    s = summarise(df, run_allocation(df, settings, settings.review_capacity_units))
    k = s[s["policy"] == "knapsack"].iloc[0].to_dict()
    m["knapsack_money_saved"] = float(k["money_saved"])
    m["knapsack_money_saved_share"] = float(k["money_saved_share"])
    return m


def run_experiment(
    settings: Settings,
    feats: pd.DataFrame,
    delays: list[float],
    seeds: list[int],
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("semi-supervised-delayed-labels")
    rows: list[dict[str, Any]] = []
    for delay in delays:
        split = make_split(feats, settings, delay)
        for seed in seeds:
            anomaly = (
                AnomalyScorer(seed).fit(feats[feats["TransactionDT"] < settings.replay_start_dt])
                if "anomaly" in methods
                else None
            )
            for method in methods:
                if delay == 0 and method in ("self_neg", "self_negpos"):
                    continue  # nothing unlabelled to pseudo-label
                with mlflow.start_run(run_name=f"{method}-d{int(delay)}-s{seed}"):
                    score, diag = run_method(method, split, seed, anomaly)
                    m = holdout_metrics(split.holdout, score, settings)
                    mlflow.log_params(
                        {
                            "method": method,
                            "label_delay_days": delay,
                            "seed": seed,
                            "n_labelled_train": len(split.train),
                            "n_val": len(split.val),
                            "n_unlabelled": len(split.unlabelled),
                        }
                    )
                    mlflow.log_metrics({**m, **diag})
                row = {
                    "method": method,
                    "delay_days": delay,
                    "seed": seed,
                    "n_labelled": len(split.train) + len(split.val),
                    "n_unlabelled": len(split.unlabelled),
                    **m,
                    **diag,
                }
                rows.append(row)
                log.info("%s d=%s s=%s pr_auc=%.4f", method, delay, seed, m["pr_auc"])
    return pd.DataFrame(rows)
