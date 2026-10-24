# Data card: IEEE-CIS Fraud Detection

The only data used by this project. It is **not** in the repository and must not be redistributed;
`data/`, `*.csv`, `*.parquet` are git-ignored and CI uses synthetic fixtures. See also the
[model card](MODEL_CARD.md). Counts below were computed from the local Kaggle files unless a
results file is cited.

## Source and terms

* **Origin:** the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle
  competition (2019). The transactions were supplied by Vesta Corporation, an e-commerce payment
  provider.
* **Licence / terms:** the data is distributed under the competition's rules and terms of use on
  Kaggle, not under an open licence. This repository does not grant any rights to it. Anyone
  reproducing the work must accept the competition terms on Kaggle themselves and read them for the
  permitted uses (the project owner should re-check the current terms before any use beyond
  portfolio/research work; this card does not summarise them). The Kaggle files stay outside the repo,
  and only aggregates and figures derived from them are published.
* **Files used:** `train_transaction.csv` and `train_identity.csv` (left join on `TransactionID`).
  The competition's `test_*` files have no labels and are not used.

## Size and splits

| item | value |
|---|---|
| transactions (`train_transaction`) | 590,540 rows, 394 columns |
| identity records (`train_identity`) | 144,233 rows, 41 columns; 24.4% of transactions have one |
| overall fraud rate | 3.50% (`isFraud`) |
| time span | `TransactionDT` is a seconds offset from an undisclosed reference; days 1 to 183 |

Time-based split (day = `TransactionDT // 86400`, 7-day purge for the label delay):

| split | days | rows | fraud rate |
|---|---|---|---|
| train | < 103 | 359,382 | 3.38% |
| validation | 110 to < 139 | 86,508 | 3.97% |
| replay (hold-out) | >= 146 | 105,693 | 3.42% |

The remaining 38,957 rows are in the purge gaps and are not used for training or evaluation. The
hold-out has 3,611 frauds (3.42%): a severe class imbalance, which is why PR-AUC and recall at a
fixed alert rate are used instead of accuracy.

## Columns

* **Used (80 model features):** `TransactionAmt`, `ProductCD`, `card1`-`card6`, `addr1`/`addr2`,
  `dist1`/`dist2`, `P_emaildomain`/`R_emaildomain`, `C1`-`C14`, `D1`-`D15`, `M1`-`M9`, `DeviceType`,
  a few `id_*` columns (`id_01`, `id_02`, `id_05`, `id_06`, `id_11`, `id_12`, `id_15`, `id_16`,
  `id_28`, `id_29`, `id_31`, `id_35` to `id_38`), derived features and 7 windowed pseudo-identity
  features. Definitions: `src/fraudrt/features/definitions.py`.
* **Anonymised columns:** the 339 `V*` columns are engineered by Vesta and their meaning is not
  disclosed; `C*`, `D*` (counts and time deltas) and `M*` (match flags) are only loosely described
  in the competition discussion, and `card1`-`card6`, `addr*`, `id_*` are masked or encoded. The
  project cannot explain these fields beyond what the data shows, and so cannot explain
  predictions in business terms. The `V*` block and most `id_*` columns are **not used**.
* **Missingness is informative and structural:** identity columns exist only for 24.4% of
  transactions; many `V*`, `D*`, `dist*` values are missing by design. "No identity record" is
  itself a strong signal in the model (the segment analysis in the model card shows recall of 16.4%
  for those rows against 66% to 74% for rows with a device type).

## Known gaps and biases

* **Label definition.** `isFraud` comes from the provider. As the competition describes it, a
  transaction is fraud when it was reported (chargeback) and the card/identity was then flagged,
  and linked transactions of the same entity were propagated. Hence labels are **chargeback-based
  and delayed**, may miss fraud nobody reported, and may label legitimate transactions of a
  compromised card as fraud. The model learns what was *reported*, not ground truth. The 7-day
  label delay in this project is an assumption; the real reporting lag is not given by the data and
  may be longer, in which case the delayed-label results would be optimistic.
* **Class imbalance.** 3.50% fraud overall, 3.42% on the hold-out. Real portfolios are often much
  lower, so absolute precision and alert-volume figures would not carry over.
* **Selection and sampling.** One provider, one channel (card-not-present e-commerce), about six
  months. Whether the data is a sample or has been down-sampled is not documented. Results do not
  generalise to other merchants, regions or in-person payments without re-validation.
* **No customer id.** The project builds a heuristic pseudo-identity from card and address fields;
  one customer can map to several identities and several customers to one.
* **Time is relative.** `TransactionDT` has no calendar date, so seasonality and holidays cannot be
  modelled or checked explicitly.
* **Dataset-start effects.** The first days have no history, which looks like drift when comparing
  to later periods (see PSI notes in the README).
* **Age.** The competition dates from 2019 and the transactions are older still (exact dates are
  not disclosed); fraud tactics and payment flows have changed since.
* **Repeat identities across splits.** 86.6% of hold-out transactions belong to a pseudo-identity
  that also appears in the train or validation days (computed with the pseudo-identity above). That
  is realistic for repeat customers, but it means the hold-out is not a test on unseen customers.

## Privacy

* The data has **no direct personal identifiers** (no names, emails, addresses, card numbers or
  phone numbers): card and address fields are masked or encoded, and the email *domain* (e.g.
  `gmail.com`) is the only email information. `DeviceType` and browser/OS strings (`id_30`,
  `id_31`, `DeviceInfo`) are device attributes.
* Even so, the masked fields combined with timestamps, amounts and devices can in principle
  single out a person; **this data should be treated as pseudonymised personal data, not anonymous
  data**, which is the safer GDPR reading. It remains subject to the Kaggle terms.
* The repository contains no raw rows, only aggregates and figures derived from the data. The
  pipeline stores transaction features and scores in a local Postgres/Redis for the replay; these
  are throw-away local containers (`compose.yaml`) holding only the dataset, not live customer data.
* GDPR notes if this were run on real data: a lawful basis (fraud prevention is commonly argued as
  legitimate interest, subject to legal assessment), purpose limitation, data minimisation (the
  design already omits unused columns), retention limits for stored scores and features, a
  DPIA for large-scale profiling, and Article 22 safeguards for any decision with significant
  effect. None of these is implemented or assessed here.

## What this project does not use

* The competition `test_transaction.csv` / `test_identity.csv` (no labels).
* The 339 `V*` columns, and `id_*` columns other than those listed above, `DeviceInfo`, `id_30`,
  and other free-text fields.
* Any external data (blacklists, geolocation, merchant data, consumer demographics).
* Any protected attribute: **the data has none**, so no fairness assessment on protected groups
  is possible. See the model card for the segment-level proxy analysis and its limits.
* Any real or live transaction data: all streaming is a replay of this file.
