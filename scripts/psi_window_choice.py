"""How large must the rolling PSI window be so that normal hold-out traffic never alerts?

Replays nothing: reads the scores the API stored in Postgres during `make replay`, slides a window
of N consecutive scores over the 37 replay days and records the distribution of score PSI against
the model's training reference. Also reports the score threshold that gives a ~3% alert rate.
Output: docs/results/psi_window.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fraudrt.config import get_settings
from fraudrt.db.repo import make_engine
from fraudrt.modeling.registry import load_from_registry
from fraudrt.monitoring.drift import feature_psi
from fraudrt.monitoring.report import load_replay_frame

s = get_settings()
bundle = load_from_registry(s)
assert bundle.reference is not None
df = load_replay_frame(make_engine(s.database_url), s.replay_start_dt).sort_values("event_dt")
scores = df["fraud_score"].to_numpy()
ref = bundle.reference["score"]
out: dict[str, object] = {
    "n_scores": len(scores),
    "model": f"{bundle.name} v{bundle.version}",
    "threshold_for_3pct_alert_rate": round(float(np.quantile(scores, 0.97)), 4),
    "alert_rate_at_configured_threshold": {
        "threshold": s.alert_score_threshold,
        "rate": round(float((scores >= s.alert_score_threshold).mean()), 4),
    },
    "windows": {},
}
for w in (1000, 2000, 5000, 10000):
    v = np.array([feature_psi(ref, scores[i - w : i]) for i in range(w, len(scores), w // 10)])
    out["windows"][str(w)] = {  # type: ignore[index]
        "positions": len(v),
        "max": round(float(v.max()), 4),
        "p99": round(float(np.quantile(v, 0.99)), 4),
        "median": round(float(np.median(v)), 4),
        "share_above_0.1": round(float((v > 0.1).mean()), 4),
    }
Path("docs/results/psi_window.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
