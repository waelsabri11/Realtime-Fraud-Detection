"""LightGBM training with MLflow tracking and model registration."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraudrt.config import Settings
from fraudrt.data import load_features, split_masks
from fraudrt.features.definitions import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from fraudrt.features.encoding import Vocab
from fraudrt.modeling.metrics import evaluate
from fraudrt.monitoring.drift import build_reference, build_score_reference

log = logging.getLogger(__name__)


def default_params(seed: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 255,
        "cat_smooth": 20.0,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 8,
        "verbosity": -1,
    }


def fit_booster(
    train: pd.DataFrame,
    val: pd.DataFrame,
    vocab: Vocab,
    params: dict[str, Any],
    max_rounds: int = 3000,
    patience: int = 100,
) -> lgb.Booster:
    xt, xv = vocab.encode_frame(train), vocab.encode_frame(val)
    cat = list(CATEGORICAL_FEATURES)
    dtrain = lgb.Dataset(xt, train["isFraud"], categorical_feature=cat, free_raw_data=False)
    dval = lgb.Dataset(xv, val["isFraud"], categorical_feature=cat, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(100)],
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def log_and_register(
    settings: Settings,
    booster: lgb.Booster,
    vocab: Vocab,
    params: dict[str, Any],
    metrics: dict[str, float],
    reference: dict[str, Any],
    tags: dict[str, str],
    promote: bool,
    run_name: str,
) -> str:
    import mlflow
    import mlflow.lightgbm
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"git_sha": _git_sha(), **tags})
        mlflow.log_params({**params, "best_iteration": booster.best_iteration})
        mlflow.log_params(
            {
                "train_end_day": settings.train_end_day,
                "replay_start_day": settings.replay_start_day,
                "label_delay_days": settings.label_delay_days,
                "n_features": len(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {"feature_columns": FEATURE_COLUMNS, "vocab": vocab.to_json()}, "spec/spec.json"
        )
        mlflow.log_dict(reference, "spec/reference.json")
        imp = pd.Series(booster.feature_importance("gain"), index=FEATURE_COLUMNS)
        mlflow.log_dict(
            {str(k): float(v) for k, v in imp.sort_values(ascending=False).round(1).items()},
            "spec/importance.json",
        )
        mlflow.lightgbm.log_model(booster, "model", registered_model_name=settings.model_name)
        run_id = run.info.run_id
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{settings.model_name}' and run_id='{run_id}'")
    version = str(max(int(v.version) for v in versions))
    if promote:
        client.set_registered_model_alias(settings.model_name, "champion", version)
    log.info("registered %s v%s (promoted=%s)", settings.model_name, version, promote)
    return version


def train_from_frames(
    settings: Settings,
    feats: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    holdout_mask: pd.Series | None,
    *,
    promote: bool,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> tuple[str, dict[str, float]]:
    train, val = feats[train_mask], feats[val_mask]
    vocab = Vocab.fit(train)
    params = default_params(settings.seed)
    booster = fit_booster(train, val, vocab, params)

    metrics: dict[str, float] = {}
    pv = booster.predict(vocab.encode_frame(val), num_iteration=booster.best_iteration)
    for k, v in evaluate(
        val["isFraud"].to_numpy(), np.asarray(pv), val["TransactionAmt"].to_numpy()
    ).items():
        metrics[f"val_{k}"] = v
    if holdout_mask is not None and holdout_mask.any():
        ho = feats[holdout_mask]
        ph = booster.predict(vocab.encode_frame(ho), num_iteration=booster.best_iteration)
        for k, v in evaluate(
            ho["isFraud"].to_numpy(), np.asarray(ph), ho["TransactionAmt"].to_numpy()
        ).items():
            metrics[f"holdout_{k}"] = v
    metrics["n_train"] = float(len(train))
    metrics["n_val"] = float(len(val))

    reference = {
        "features": build_reference(vocab.encode_frame(train)),
        "score": build_score_reference(np.asarray(pv)),
    }
    version = log_and_register(
        settings,
        booster,
        vocab,
        params,
        metrics,
        reference,
        {"seed": str(settings.seed), **(tags or {})},
        promote,
        run_name,
    )
    return version, metrics


def run_training(settings: Settings, promote: bool = True) -> tuple[str, dict[str, float]]:
    feats = load_features(settings)
    m = split_masks(feats, settings)
    return train_from_frames(
        settings,
        feats,
        m["train"],
        m["val"],
        m["replay"],
        promote=promote,
        run_name="train-initial",
        tags={"kind": "initial"},
    )
