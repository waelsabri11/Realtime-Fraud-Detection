"""Figures + frozen result JSONs for the README (reads work/results/*.json)."""

import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, DST = Path("work/results"), Path("docs/results")
DST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
COL = {
    "topk_score": "#9a9a9a",
    "greedy_value": "#e0a030",
    "knapsack": "#1f6fb4",
    "oracle": "#2a9d5b",
}

alloc = json.loads((SRC / "allocation.json").read_text())
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for pol in ("topk_score", "knapsack", "oracle"):
    pts = sorted(
        (r["capacity_units_per_day"], r["money_saved_share"] * 100)
        for r in alloc["capacity_sweep"]
        if r["policy"] == pol
    )
    ax.plot(*zip(*pts, strict=True), marker="o", color=COL[pol], label=pol.replace("_", " "))
ax.set_xscale("log")
ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("analyst review capacity (units / day)")
ax.set_ylabel("% of fraud $ stopped")
ax.set_title("Alert allocation on the replay period")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(DST / "allocation.png", dpi=160)

mon = json.loads((SRC / "monitoring.json").read_text())
top = mon["drift"]["top_features"][:12][::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.barh([t["feature"] for t in top], [t["psi"] for t in top], color="#1f6fb4")
ax.axvline(0.1, color="#e0a030", ls="--", lw=1)
ax.axvline(0.25, color="#c0392b", ls="--", lw=1)
ax.set_xlabel("PSI vs training distribution (replay period)")
ax.set_title("Most drifted features")
fig.tight_layout()
fig.savefig(DST / "drift_features.png", dpi=160)

perf = [r for r in mon["asof_performance"] if r["pr_auc_known_labels"] is not None]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(
    [r["monitor_day"] for r in perf],
    [r["pr_auc_known_labels"] for r in perf],
    marker="o",
    color="#1f6fb4",
    label="PR-AUC from labels known at that day",
)
ax.plot(
    [r["monitor_day"] for r in perf],
    [r["pr_auc_eventual"] for r in perf],
    marker="s",
    color="#9a9a9a",
    label="same window, once all labels are in",
)
ax.set_xlabel("monitoring day")
ax.set_ylabel("PR-AUC (14-day window)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(DST / "delayed_label_monitoring.png", dpi=160)

for f in [
    *SRC.glob("loadtest_*.json"),
    *[
        SRC / f"{n}.json"
        for n in (
            "training",
            "allocation",
            "monitoring",
            "retrain",
            "parity",
            "label_gate",
            "replay",
        )
    ],
]:
    if f.exists():
        shutil.copy(f, DST / f.name)
print("figures written to", DST)
