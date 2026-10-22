"""Retrain with the same seed/data and compare predictions with the registered champion."""

import json
from pathlib import Path

import numpy as np

from fraudrt.config import get_settings
from fraudrt.data import load_features, split_masks
from fraudrt.modeling.registry import load_from_registry
from fraudrt.modeling.train import default_params, fit_booster

s = get_settings()
feats = load_features(s)
m = split_masks(feats, s)
champion = load_from_registry(s)
train, val = feats[m["train"]], feats[m["val"]]
booster = fit_booster(train, val, champion.vocab, default_params(s.seed))
x = champion.vocab.encode_frame(val)
a = np.asarray(booster.predict(x, num_iteration=booster.best_iteration))
b = np.asarray(champion.booster.predict(x))
res = {
    "champion_version": champion.version,
    "best_iteration_retrained": booster.best_iteration,
    "best_iteration_champion": champion.booster.best_iteration,
    "max_abs_prediction_diff": float(np.abs(a - b).max()),
    "identical": bool(np.array_equal(a, b)),
}
print(json.dumps(res, indent=2))
Path("work/results").mkdir(parents=True, exist_ok=True)
Path("work/results/reproducibility.json").write_text(json.dumps(res, indent=2))
