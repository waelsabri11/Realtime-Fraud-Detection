<!-- Dry-run output of the alert receiver (ALERT_DRY_RUN, the default): this is the body that would be
posted as a GitHub issue. It comes from run 1 of the `email_domain_all_protonmail` scenario in
results/alerting.json (every transaction's P_emaildomain set to protonmail.com from transaction 7,000).
No issue was created on GitHub. "n/a" for the 5xx ratio means no 5xx response existed at all. The trailing comment is what the receiver appends when the alert resolves. -->

# [alert] ScorePSISignificant: score distribution drifted (PSI above 0.25)

labels: monitoring, alert:ScorePSISignificant

---

<!-- fraudrt-alert:ScorePSISignificant -->
**Alert `ScorePSISignificant`** (critical) fired at `2026-10-27T11:49:50.63Z`; this issue was opened at `2026-10-27T11:49:55.660Z`.

score distribution drifted (PSI above 0.25)

PSI of the last 5000 scores against the training reference is above 0.25: a significant shift, consider retraining.

### Values

* value that triggered the rule: **0.349**
* rule window: rolling window of 5000 transactions, for 10s
* time window of interest: `2026-10-27T11:49:50.63Z` to `2026-10-27T11:49:55.660Z`

### Readings when the alert arrived

| metric | value |
|---|---|
| score PSI (rolling window) | 0.3345 |
| alert rate (share of scored transactions) | 0.03531 |
| p99 /score latency (s) | 0.003911 |
| 5xx ratio | n/a |
| consumer lag, transactions topic | 28 |
| requests per second | 100.4 |

### Features with the highest PSI now (rolling window vs training reference)

| feature | PSI |
|---|---|
| `P_emaildomain` | 16.081 |
| `uid_secs_since_last` | 1.951 |
| `email_match` | 1.563 |
| `id_31` | 0.796 |
| `D11` | 0.606 |

Several features sit above 0.25 even in normal operation (see `docs/MODEL_CARD.md`), so the level alone is not the signal; see the largest increase below. `hour` and `dow` are left out: a window shorter than a week never matches their reference.

### Largest PSI increase since `2026-10-27T11:44:50Z`

| feature | before | now |
|---|---|---|
| `P_emaildomain` | 4.470 | 16.081 |
| `uid_secs_since_last` | 0.078 | 1.951 |
| `email_match` | 0.187 | 1.563 |
| `uid_sum_24h` | 0.019 | 0.076 |
| `uid_cnt_24h` | 0.021 | 0.074 |

[Dashboard](http://localhost:3000/d/fraud-monitoring/fraud-monitoring?from=1793100890630&to=1793102095660)
[Rule expression in Prometheus](http://4f95eb8a4fe6:9090/graph?g0.expr=fraud_score_psi+%3E+0.25&g0.tab=1)
