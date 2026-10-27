# Model card: fraud scoring model (`fraud-lgbm`)

Status: portfolio / research system. It has **never** scored live payments; "real time" is a replay
of the IEEE-CIS data (see the [README](../README.md)). Every number below comes from a file in
[`docs/results/`](results) or the README, which come from the same run. See also the
[data card](DATA_CARD.md).

## 1. Intended use

* **Decision support for fraud analysts, with a human in the loop.** The model gives each
  transaction a fraud probability. A daily review queue is built from the scores under a fixed
  analyst capacity (100 review units per simulated day; see "Alert allocation" in the README).
  An analyst reviews the queue and decides.
* **Not intended for automatic payment blocking, declining or account closure.** A score alone must
  not trigger an adverse action against a customer. The alert rates used below (1% to 3% of
  transactions) are review-queue sizes, not decline rates.

### Out-of-scope uses

* Automatic decisions on real customers, or any use without a human reviewing the case.
* Credit, insurance, pricing, employment or any other decision than card-not-present-style fraud
  review. The model was not designed or validated for them.
* Use on data from another merchant, processor, country or period without re-validation. It was
  trained on one dataset (one processor, about six months, see the data card).
* Any claim about fairness on protected groups (see section 6).
* Treating the allocation figures as business results: they rest on assumed review costs.

## 2. Model details

| item | value |
|---|---|
| algorithm | LightGBM, binary objective, deterministic, seed 42 |
| features | 80: raw IEEE-CIS columns (without the 339 `V*` block and most `id_*` columns), 5 derived (hour, day of week, log amount, amount cents, email match) and 7 windowed pseudo-identity features |
| pseudo-identity | `card1..card6 + addr1 + P_emaildomain` (there is no customer id in the data) |
| trees | 378, chosen by early stopping on validation PR-AUC |
| tuning | none; hyper-parameters are fixed defaults in `modeling/train.py` |
| reproducibility | retraining with the same seed reproduces the registered model's predictions bit for bit (`scripts/check_reproducibility.py`, last run `identical: true`, max abs difference 0.0; its JSON output is in the git-ignored `work/` folder, not in `docs/results/`) |
| serving | FastAPI `/score`, model loaded from the MLflow registry (alias `champion`) |

Split by day of `TransactionDT`, with a 7-day purge for the label delay (so no training label is
one that would not yet have been known at training time):

| split | days | rows | fraud rate |
|---|---|---|---|
| train | < 103 | 359,382 | 3.38% |
| validation (early stopping only) | 110 to < 139 | 86,508 | 3.97% |
| replay (hold-out, never tuned on) | >= 146 | 105,693 | 3.42% |

The remaining rows of the 590,540 in the file fall in the two purge gaps. Labels are released
`LABEL_DELAY_DAYS = 7` simulated days after the transaction.

## 3. Performance (as measured)

Source: [`training.json`](results/training.json) and README "Results". Replay hold-out, 105,693
transactions, 3,611 frauds. One split, one training run, no confidence intervals.

| metric | validation (early-stopped on it) | **replay hold-out** |
|---|---|---|
| PR-AUC | 0.598 | **0.490** |
| ROC-AUC | 0.921 | 0.898 |
| recall, alerting 1% of transactions | 23.0% (precision 91%) | 24.1% (precision 82%) |
| recall, alerting 3% of transactions | 50.3% (precision 67%) | 46.1% (precision 52%) |
| fraud $ stopped, alerting 3% | 45.1% | 37.0% |

Base rate on the hold-out is 3.42%, so precision at 3% alerts (52%) is about 15 times the base rate
but nearly half of the alerts are legitimate transactions (false positive rate 1.48% of legitimate
transactions, see section 4).

**Allocation under a fixed review capacity** ([`allocation.json`](results/allocation.json), 37 replay
days, 3,611 frauds worth $539,698, 100 units/day, cases over $500 cost 2 units; a reviewed fraud is
assumed fully stopped):

| policy | reviews | frauds caught | **fraud $ stopped** | share of fraud $ |
|---|---|---|---|---|
| top-k by score | 3,512 | 1,686 | $199,873 | 37.0% |
| knapsack (exact DP) | 3,097 | 1,203 | $272,379 | 50.5% |

The knapsack stops more money while catching fewer fraudulent transactions: it prefers large
amounts. Greedy by p x amount ($284,485) is equivalent within noise here. Both the capacity and the
cost model are assumptions.

**Latency.** Server-side handler time during the replay (features, Redis, model; excludes network
and JSON parsing): p50 0.41 ms, p95 0.70 ms, **p99 1.16 ms**. Locust, 1 worker, 4 users: p99 5 ms at
1,287 req/s; 32 users (saturated): p99 72 ms at 1,354 req/s (client side, loopback, one laptop).

**Online/offline parity.** 100% of numeric and categorical feature cells served by the API equal the
offline table for all 105,693 replayed transactions ([`parity.json`](results/parity.json)).

## 4. Segment error analysis (new, proxy only)

Script: [`scripts/segment_errors.py`](../scripts/segment_errors.py) (logic in
`src/fraudrt/monitoring/segments.py`). Output: [`segment_errors.json`](results/segment_errors.json).
It retrains the model offline with the same seed and checks that the hold-out PR-AUC equals the
value in `training.json` before writing anything. Alerts are the **global** top 3% of scores on the
replay hold-out (3,171 alerts): overall recall 46.1%, precision 52.4%, false positive rate 1.48%.
FPR = share of *legitimate* transactions that are alerted. The cut-off is global, not per segment,
so segments with a higher fraud rate get more alerts.

| segment | n | fraud rate | alert rate | recall | precision | FPR |
|---|---|---|---|---|---|---|
| **ProductCD** C | 10,988 | 13.3% | 15.8% | 74.2% | 62.5% | 6.83% |
| H | 3,017 | 6.1% | 8.1% | 54.6% | 41.2% | 5.08% |
| R | 4,983 | 4.9% | 5.5% | 74.0% | 65.6% | 1.98% |
| S | 3,310 | 5.5% | 2.7% | 35.9% | 73.9% | 0.74% |
| W | 83,395 | 1.9% | 1.0% | 15.2% | 28.2% | 0.73% |
| **card network** visa | 68,410 | 3.5% | 2.8% | 43.2% | 53.2% | 1.38% |
| mastercard | 34,385 | 3.2% | 3.1% | 52.0% | 53.4% | 1.48% |
| discover | 1,134 | 8.2% | 9.9% | 40.9% | 33.9% | 7.11% |
| american express | 1,020 | 2.9% | 4.0% | 60.0% (30 frauds) | 43.9% | 2.32% |
| missing | 744 | 2.0% | 2.8% | 66.7% (15 frauds, low support) | 47.6% | 1.51% |
| **card type** credit | 24,474 | 6.4% | 6.8% | 57.4% | 53.4% | 3.39% |
| debit | 80,476 | 2.5% | 1.8% | 37.3% | 51.4% | 0.92% |
| missing | 743 | 2.0% | 2.8% | 66.7% (15 frauds, low support) | 47.6% | 1.51% |
| **device type** desktop | 12,424 | 7.9% | 9.4% | 73.9% | 61.6% | 3.94% |
| mobile | 8,755 | 11.7% | 12.5% | 66.3% | 61.7% | 5.43% |
| missing (no identity record) | 84,514 | 1.9% | 1.1% | 16.4% | 29.3% | 0.77% |
| **amount ($)** < 25 | 7,708 | 7.6% | 9.7% | 69.7% | 54.8% | 4.75% |
| 25-50 | 26,113 | 3.2% | 2.3% | 48.7% | 65.5% | 0.83% |
| 50-100 | 28,468 | 2.7% | 2.0% | 37.1% | 50.3% | 1.03% |
| 100-250 | 31,903 | 2.7% | 2.4% | 39.2% | 44.8% | 1.35% |
| 250-500 | 7,012 | 4.6% | 4.4% | 49.1% | 51.1% | 2.24% |
| >= 500 | 4,489 | 5.2% | 3.8% | 28.5% | 39.2% | 2.44% |

Reading it:

* **Recall and FPR vary a lot across segments** (recall 15% to 74%, FPR 0.7% to 7%), mostly because
  the fraud rate varies a lot: the model alerts where it sees risk. Legitimate customers in
  `ProductCD = C`, mobile, credit-card and sub-$25 transactions are alerted 4% to 7% of the time,
  against about 0.7% to 1.5% elsewhere. In a human-in-the-loop queue this means those customers are
  *reviewed* more often, which is a cost (friction, analyst time) and a place to monitor.
* **The model is weakest where most of the volume is.** `ProductCD = W` and rows with no identity
  record (`DeviceType` missing) are each about 79% to 80% of transactions and hold 43% and 45% of the frauds, but
  recall is only 15% to 16%. Those two segments largely overlap; the overlap was not computed.
* **Large amounts (>= $500) have the lowest recall of the amount buckets (28.5%)**: the model score
  does not weight amount, the allocation step does (the knapsack above).
* `ProductCD`, `card4`, `card6` and `DeviceType` are model inputs, so the model is free to use them.
* Segment definitions: `ProductCD` is an anonymised product code (its meaning is not documented
  by the data provider); "missing" for `DeviceType` means the transaction has no row in the identity
  table (75.6% of the 590,540 rows in the file), not an unknown device. Segments with fewer than 1,000 rows or
  30 frauds carry a `low_support` flag in the JSON; 30 frauds is still a wide interval, so American
  Express and Discover figures are indicative only. No confidence intervals were computed.

**This is not a fairness audit.** IEEE-CIS contains no protected attributes (no age, sex, ethnicity,
religion, nationality, disability) and no customer demographics. **Fairness on protected groups
cannot be assessed with this data.** The segment disparities above are a proxy: they show where error
rates differ by product, payment instrument, device and amount, and they say nothing about
whether any protected group is treated differently. The pseudo-identity and fields such as
`addr1` or the email domain could correlate with protected attributes in a real deployment; that
could not be tested here.

## 5. Limitations

* One dataset, one split, one run; no confidence intervals on any figure above. The semi-supervised
  experiments (README) use 5 seeds, but the headline model is a single run.
* The hold-out PR-AUC (0.490) is clearly below the validation figure (0.598): part of the gap is
  the shift between periods, part is the early stopping on validation.
* Features are deliberately light (no `V*` block, no text fields, no tuning), so the PR-AUC is below
  what a heavily engineered Kaggle solution reaches on a random split. The system is the point.
* The 7-day label delay and the review cost model are assumptions, not properties of the data.
* Labels are chargeback-based and delayed (see the data card); the "truth" the model learns is what
  was eventually reported, not what was fraud.
* The identity is a heuristic; one customer can map to several identities.
* PSI is computed against the training distribution, so dataset-start effects register as drift.
* The delayed-label performance estimate is noisy (about 620 to 770 frauds per 14-day window).
* The anomaly score from the semi-supervised study is **not** served: it did not improve PR-AUC
  outside noise.

## 6. Monitoring

Implemented in `src/fraudrt/monitoring/` ([`monitoring.json`](results/monitoring.json)):

* **Label-free, immediate:** PSI of the score and of each served feature against the training
  distribution (conventional reading: < 0.1 stable, 0.1 to 0.25 moderate, >= 0.25 significant).
  Measured on the replay: score PSI 0.0008 overall (weekly 0.0006 to 0.009); 8 of 80 features at
  PSI >= 0.25 (largest `id_31` 0.63, `M7`/`M8`/`M9` 0.44, `D11` 0.41, `M1`-`M3` 0.36) and 3 in
  0.1 to 0.25 (`uid_secs_since_last` 0.17, `D15`, `id_38`).
* **Delayed-label performance:** every 7 days, PR-AUC over the last 14 days using only labels known
  that day (about half of the window) vs. the same transactions once all labels are in
  (README table; e.g. day 160: 0.456 vs 0.501).
* **What would alert first:** the label-free PSI, because labels are a week late and noisy. In the
  measured run the score PSI stayed flat while several feature PSIs were high from the start, so a
  fixed feature-PSI threshold is too noisy on its own (it would fire from day one): the score PSI and
  the *change* in feature PSI are the more useful signals.
* **Live monitoring and alerting (opt-in, `make monitoring-up`; README section "Live monitoring and
  alerting"):** Prometheus metrics from the API and the stream worker, a Grafana dashboard, and
  Alertmanager rules for p99 latency, 5xx rate, consumer lag and the rolling score PSI (> 0.1 and
  > 0.25, window of the last 5,000 scores; 2,000 gave a false alarm on normal hold-out traffic, see
  [`psi_window.json`](results/psi_window.json)). A firing alert opens one GitHub issue per alert name
  through a webhook receiver (dry-run by default; no real issue has been filed).
  Measured on a replay at 100 transactions/s with injected drift
  ([`alerting.json`](results/alerting.json)): a 100% switch of the email domain was alerted at
  PSI 0.35 with an issue 57 to 63 s after injection; an amount shift of x3 or x5 (PSI 0.14 to 0.17)
  raised the 0.1 alert with an issue 77 to 83 s after injection; **x2 amount and a 50% domain shift
  (PSI 0.08 to 0.09) were not detected**. These are synthetic shifts, so they bound the pipeline's
  latency and sensitivity, not the model's behaviour under real drift.
* **Still not implemented:** segment-level monitoring over time, calibration monitoring and
  analyst-override-rate tracking (the alert *volume* is exported and charted, with no alert rule on
  it), a paging channel other than the GitHub issue, and any action on an alert: nothing retrains or
  rolls back automatically. These are the ones to add first in a real deployment, together with
  re-running the segment table above on every retrain.

## 7. Retraining

`make retrain` simulates a retrain on day 165 using only history plus replayed transactions
*as served* whose label had arrived (498,113 rows, 13,266 from the replay period). The challenger is
registered and **not** promoted. Evaluated on the 49,630 transactions of days 166 to 182
([`retrain.json`](results/retrain.json)):

| model | PR-AUC | recall @ 3% alerts | fraud $ stopped @ 3% alerts |
|---|---|---|---|
| champion v1 | 0.491 | 44.3% | $107,525 (35.8%) |
| challenger v2 | 0.549 | 48.9% | $127,877 (42.5%) |

A single split, no confidence intervals. Promotion is a manual decision; no cadence or automatic
trigger is defined. Under a label delay, accuracy degrades with the age of the training data
(semi-supervised study: hold-out PR-AUC 0.515 with no delay to 0.461 with a 60-day delay), and the
semi-supervised methods tried did not help beyond noise.

## 8. Risks and mitigations

| risk | what could happen | mitigation here | what is missing |
|---|---|---|---|
| **False positives block or burden legitimate customers** | At 3% alerts, 47.6% of alerts are legitimate (precision 52.4%); 1.48% of legitimate transactions are alerted, up to 6.8% in `ProductCD = C` and 7.1% for Discover | Intended use is review, not automatic blocking; alert volume capped by analyst capacity; segment error table above | Customer-friction metrics, appeal / feedback path, per-segment alert-rate limits |
| **Drift** | Fraud patterns, products or devices change; the model degrades silently | PSI on score and features; live PSI alerts (> 0.1, > 0.25) with automatic issue filing; delayed-label PR-AUC; retraining with only known labels | Small shifts (score PSI < 0.1) go undetected; calibration monitoring; an agreed retrain trigger |
| **Feedback loops from analyst-reviewed labels** | Only alerted cases are investigated, so confirmed-fraud labels over-represent what the model already flags; unreviewed frauds look like negatives | In this study labels are the dataset's ground truth for **all** transactions, so the loop **is not simulated** and its effect is not measured | In production: label a random sample of non-alerted transactions, and track chargeback labels separately from analyst decisions |
| **Adversarial adaptation** | Fraudsters learn what passes (small amounts, new devices, fresh identities) and shift behaviour | Windowed identity features, retraining; low-support behaviours are visible in the segment table | No adversarial testing was done; no rate-limiting or rules layer |
| **Proxy for protected attributes** | Features such as `addr1` or the email domain may correlate with protected characteristics | None; not testable with this data | A fairness audit on real data with demographic information, under legal review |
| **Misuse as an automatic decision** | A score used as a decline rule | Documented as out of scope here | Technical controls and governance in a real deployment |

## 9. Regulatory framing (informational, not a compliance claim)

Fraud detection is generally not itself a high-risk use under the EU AI Act, but the same system
used for credit or other access-to-service decisions could be; this repo makes no compliance claim.
Under GDPR, solely automated decisions with significant effects on a person need safeguards (Article
22); the human-in-the-loop intended use keeps this system on the decision-support side. Legal
review would be needed before any real deployment.

## 10. How to reproduce

```bash
make demo                          # train, replay, parity, allocate, monitor, retrain, figures
uv run python scripts/segment_errors.py   # writes docs/results/segment_errors.json
```

Needs the Kaggle files locally (not in the repository).
