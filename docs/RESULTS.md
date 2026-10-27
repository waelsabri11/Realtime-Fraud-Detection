# Detailed results

Supporting results for the [README](../README.md); raw outputs are in [`results/`](results).

### Alert allocation under a fixed review capacity

Replay period, 37 days, 3,611 frauds worth $539,698. Capacity: 100 review units per day (a case over
$500 costs 2, others 1). "Money saved" = amount of the frauds among the reviewed cases, assuming a
reviewed fraud is stopped.

| policy | reviews | frauds caught | recall | **fraud $ stopped** | share of fraud $ | expected loss saved (model) |
|---|---|---|---|---|---|---|
| top-k by score (baseline) | 3,512 | 1,686 | 46.7% | $199,873 | 37.0% | $202,636 |
| greedy by p x amount | 2,805 | 1,066 | 29.5% | $284,485 | 52.7% | $238,708 |
| **knapsack (exact DP)** | 3,097 | 1,203 | 33.3% | **$272,379** | **50.5%** | **$242,645** |
| oracle (true outcomes) | 3,122 | 3,122 | 86.5% | $530,264 | 98.3% | n/a |

![allocation](results/allocation.png)

The knapsack stops 36% more fraud money than top-k by score at the same capacity (50.5% vs 37.0%
of fraud $), while catching fewer fraudulent transactions (1,203 vs 1,686): it spends reviews on
large amounts. It maximises *expected* loss saved, which it does (greedy is within 2%); on
realised money greedy-by-value happens to land $12k higher at this capacity, so the two should be
read as equivalent here. The gap between "greedy" and "knapsack" only exists because costs are
non-uniform; with unit costs they coincide (tested). The capacity sweep (25 to 400 units/day) is in
[`allocation.json`](results/allocation.json): knapsack beats top-k at every capacity (e.g. 24.7%
vs 12.1% of fraud $ at 25 units/day, 79.5% vs 71.4% at 400).

### Drift and label-aware monitoring

PSI of each served feature and of the score against the training distribution
([`monitoring.json`](results/monitoring.json)):

* **Score PSI 0.0008** overall (weekly values 0.0006 to 0.009): the score distribution is stable.
* 8 of 80 features have PSI >= 0.25 and 3 are between 0.1 and 0.25. Largest: `id_31` (browser
  version, 0.63), `M8`/`M9`/`M7` (0.44), `D11` (0.41), `M1`-`M3` (0.36).
* `uid_secs_since_last` (0.17, stable at 0.13 to 0.21 each week) partly reflects the dataset's cold
  start: the first training days have no identity history, so "first seen" is over-represented in
  the reference. The other windowed identity features have weekly PSI < 0.05.

![drift](results/drift_features.png)

**Delayed labels.** With a 7-day delay, a monitoring job on day T can only score transactions up to
day T − 7. `asof_performance` evaluates, every 7 days, the last 14 days of traffic using only labels
known at that day (about half of that window, 49% to 53%) and compares with the same transactions
once all labels are in:

| monitoring day | labels known / scored | PR-AUC from known labels | PR-AUC with all labels |
|---|---|---|---|
| 153 | 0 / 20,263 | n/a | 0.456 |
| 160 | 20,263 / 41,230 | 0.456 | 0.501 |
| 167 | 20,967 / 41,769 | 0.540 | 0.498 |
| 174 | 20,802 / 41,191 | 0.453 | 0.468 |
| 181 | 20,389 / 38,168 | 0.490 | 0.500 |

![delayed labels](results/delayed_label_monitoring.png)

The delayed estimate is noisy (about 620 to 770 frauds per window) and always about a week stale;
that is why the label-free score/feature PSI is what would alert first.

### Retraining with only the labels known at the time

`make retrain` simulates retraining on day 165 (RETRAIN_DAY): training data = history plus replayed
transactions *as served* (features exactly as the model saw them) whose label had arrived by day 165:
498,113 training rows (13,266 from the replay period) and 22,071 replay rows for early stopping. The
challenger is registered as a new model version and **not** promoted. Evaluated on the replay days
after the retraining date (days 166 to 182, 49,630 transactions, all labels):

| model | PR-AUC | ROC-AUC | recall @ 3% alerts | fraud $ stopped @ 3% alerts |
|---|---|---|---|---|
| champion v1 (trained on days < 103) | 0.491 | 0.898 | 44.3% | $107,525 (35.8%) |
| challenger v2 (day-165 retrain) | 0.549 | 0.920 | 48.9% | $127,877 (42.5%) |

A single split, no confidence intervals; the champion was about two months stale at that point.

### Semi-supervised learning under delayed labels

Question: when labels arrive late, can the unlabelled recent transactions help? At the retraining
time T (start of the replay period) and a label delay *d*, only transactions up to day T - d have a
label; the *d* most recent days are unlabelled. `fraudrt semisup` trains each method on that
state and evaluates on the same hold-out as above (days >= 146, 105,693 transactions), 5 seeds per
cell, every run tracked in MLflow (experiment `semi-supervised-delayed-labels`). Early stopping uses
the newest 21 labelled days. Offline experiment: it does not go through the stream.

* **supervised** - LightGBM on the labelled part (baseline).
* **self_neg / self_negpos** - one round of pseudo-labelling of the unlabelled window with the
  baseline, isotonic-calibrated on validation: negatives below p = 0.01; for `self_negpos` also
  positives above p = 0.7, capped at half of the expected frauds in that window (guards against
  confirmation bias).
* **pu** - Elkan-Noto PU learning: known frauds vs everything else, scores divided by the estimated
  label frequency c.
* **anomaly** - Isolation Forest on the numeric features, fitted on all rows before T without labels,
  added as a feature (`features/anomaly.py`).

PR-AUC on the hold-out, mean +/- std over 5 seeds:

| delay (labelled / unlabelled rows) | supervised | self_neg | self_negpos | pu | anomaly |
|---|---|---|---|---|---|
| 0 d (all labelled) | **0.515 +/- 0.003** | n/a | n/a | 0.456 +/- 0.006 | 0.515 +/- 0.003 |
| 7 d | 0.506 +/- 0.005 | 0.504 +/- 0.003 | **0.508 +/- 0.004** | 0.434 +/- 0.008 | 0.503 +/- 0.002 |
| 30 d | **0.485 +/- 0.003** | 0.476 +/- 0.004 | 0.481 +/- 0.005 | 0.378 +/- 0.008 | 0.481 +/- 0.005 |
| 60 d | 0.461 +/- 0.004 | 0.449 +/- 0.004 | 0.458 +/- 0.003 | 0.333 +/- 0.011 | **0.463 +/- 0.003** |

![semi-supervised](results/semisup.png)

What the runs say (full table incl. ROC-AUC, recall at 3% alerts, knapsack money saved, cold-start
PR-AUC: [`semisup_summary.csv`](results/semisup_summary.csv), per-run:
[`semisup_runs.csv`](results/semisup_runs.csv)):

* **Delay hurts, and that is the real effect:** PR-AUC drops from 0.515 (d = 0) to 0.461 (d = 60),
  and fraud $ stopped by the knapsack from 51.9% to 44.4%.
* **None of the semi-supervised methods beats the supervised baseline outside noise.** Self-training
  is equal or slightly worse (-0.012 at d = 60 for negatives only). The pseudo-labels are decent
  (pseudo-negatives contain 0.5% to 0.6% frauds against a ~3.4% base rate; capped
  pseudo-positives are 77% to 83% precise, see
  [`semisup_diagnostics.csv`](results/semisup_diagnostics.csv)) but they carry no information
  the baseline did not already have (a plausible reading, not tested separately).
* **PU learning is clearly worse** (0.33 to 0.46). Here the labelled negatives are genuine, so
  treating them as unlabelled throws information away; the estimated label frequency c falls from
  0.37 to 0.09 as the delay grows, i.e. PU's correction is driven by the unlabelled window. PU would
  be the right framing only if many "negatives" were hidden frauds, which this dataset cannot show.
* **Anomaly score:** +0.002 PR-AUC at d = 60 (about one standard deviation), 0 or slightly negative
  elsewhere; cold-start PR-AUC (first-seen identities) is also within noise (0.534 vs 0.530 at d = 60).
  Not significant, so **it is not served by `/score`**. It is implemented as a shared feature
  definition with batch/online equality tested (`test_anomaly_score_batch_equals_online_row`), so
  wiring it in later needs only a model trained with it.

Caveats: one split and one retrain time; the retraining time here (day 146) is not the stream
retrain of the previous section; a small autoencoder was not tried; thresholds (0.01 / 0.7 / cap)
were fixed in advance and not tuned.

