"""Summary table + plots for the semi-supervised experiment (reads work/results/semisup_runs.csv)."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
SRC, DST = Path("work/results"), Path("docs/results")
runs = pd.read_csv(SRC / "semisup_runs.csv")
metrics = ["pr_auc", "roc_auc", "recall_at_3pct", "cold_start_pr_auc", "knapsack_money_saved_share"]
g = runs.groupby(["delay_days", "method"])[metrics].agg(["mean", "std"])
g.columns = [f"{a}_{b}" for a, b in g.columns]
g["n_seeds"] = runs.groupby(["delay_days", "method"]).size()
g["n_labelled"] = runs.groupby(["delay_days", "method"])["n_labelled"].first()
g["n_unlabelled"] = runs.groupby(["delay_days", "method"])["n_unlabelled"].first()
g = g.reset_index()
g.round(5).to_csv(DST / "semisup_summary.csv", index=False)
diag = runs[runs["method"].isin(["self_neg", "self_negpos", "pu"])].groupby(
    ["delay_days", "method"]
)
extra = diag[[c for c in runs.columns if c.startswith("pseudo_") or c == "elkan_noto_c"]].mean()
extra.round(4).reset_index().to_csv(DST / "semisup_diagnostics.csv", index=False)

COL = {
    "supervised": "#1f6fb4",
    "self_neg": "#e0a030",
    "self_negpos": "#c0392b",
    "pu": "#9a9a9a",
    "anomaly": "#2a9d5b",
}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (m, title) in zip(
    axes,
    [
        ("pr_auc", "PR-AUC on the replay hold-out"),
        ("knapsack_money_saved_share", "Fraud $ stopped (knapsack, 100 units/day)"),
    ],
    strict=True,
):
    for method, col in COL.items():
        d = g[g["method"] == method]
        ax.errorbar(
            d["delay_days"],
            d[f"{m}_mean"],
            yerr=d[f"{m}_std"],
            marker="o",
            color=col,
            label=method.replace("_", " "),
            capsize=3,
        )
    ax.set_xlabel("label delay (days)")
    ax.set_title(title)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(DST / "semisup.png", dpi=160)
(DST / "semisup_summary.json").write_text(
    json.dumps(json.loads(g.round(5).to_json(orient="records")), indent=1)
)
print(
    g[
        [
            "delay_days",
            "method",
            "pr_auc_mean",
            "pr_auc_std",
            "recall_at_3pct_mean",
            "cold_start_pr_auc_mean",
            "knapsack_money_saved_share_mean",
            "n_seeds",
        ]
    ]
    .round(4)
    .to_string()
)
