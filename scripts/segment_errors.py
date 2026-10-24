"""Recall / precision / FPR at a 3% alert rate by segment on the replay hold-out.

Retrains the champion recipe offline (same seed, data and split as `fraudrt train`, no MLflow),
checks the hold-out PR-AUC against docs/results/training.json, then writes
docs/results/segment_errors.json.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from fraudrt.config import get_settings
from fraudrt.data import load_features, split_masks
from fraudrt.features.encoding import Vocab
from fraudrt.modeling.train import default_params, fit_booster
from fraudrt.monitoring.segments import segment_report

SEGMENT_COLUMNS = ["ProductCD", "card4", "card6", "DeviceType"]

s = get_settings()
feats = load_features(s)
m = split_masks(feats, s)
train, val, hold = feats[m["train"]], feats[m["val"]], feats[m["replay"]].reset_index(drop=True)
vocab = Vocab.fit(train)
booster = fit_booster(train, val, vocab, default_params(s.seed))
score = np.asarray(booster.predict(vocab.encode_frame(hold), num_iteration=booster.best_iteration))

pr_auc = float(average_precision_score(hold["isFraud"], score))
ref = json.loads(Path("docs/results/training.json").read_text())["holdout_pr_auc"]
assert abs(pr_auc - ref) < 1e-9, f"hold-out PR-AUC {pr_auc} != recorded {ref}"

report = segment_report(hold, score, SEGMENT_COLUMNS, alert_rate=0.03)
report["model"] = {
    "best_iteration": int(booster.best_iteration),
    "holdout_pr_auc": pr_auc,
    "matches_training_json": True,
}
Path("docs/results/segment_errors.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["overall"], indent=2))
