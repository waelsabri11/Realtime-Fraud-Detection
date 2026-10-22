from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from fraudrt.config import Settings
from fraudrt.data import build_features, split_masks
from fraudrt.db.models import Base
from fraudrt.db.repo import Repository
from fraudrt.modeling.metrics import evaluate
from fraudrt.monitoring.drift import (
    build_reference,
    build_score_reference,
    feature_psi,
    psi_from_props,
)
from fraudrt.monitoring.performance import asof_series, known_at
from fraudrt.streaming.replayer import build_events
from fraudrt.streaming.worker import LabelGate
from fraudrt.synthetic import make_synthetic


def test_split_has_label_delay_gap() -> None:
    s = Settings(train_end_day=20, replay_start_day=30, label_delay_days=7, _env_file=None)
    feats = build_features(make_synthetic(n=3000, days=40))
    m = split_masks(feats, s)
    assert feats.loc[m["train"], "TransactionDT"].max() < (20 - 7) * 86_400
    assert feats.loc[m["val"], "TransactionDT"].max() < (30 - 7) * 86_400
    assert feats.loc[m["replay"], "TransactionDT"].min() >= 30 * 86_400
    assert not (m["train"] & m["val"]).any()
    assert not (m["val"] & m["replay"]).any()


def test_replay_events_are_time_ordered_and_labels_delayed() -> None:
    s = Settings(replay_start_day=25, label_delay_days=7, _env_file=None)
    raw = make_synthetic(n=2000, days=40)
    events = build_events(raw, s)
    times = [e[0] for e in events]
    assert times == sorted(times)
    tx = {json.loads(e[4])["transaction_id"]: e[0] for e in events if e[2] == s.topic_transactions}
    for e in events:
        if e[2] == s.topic_labels:
            lab = json.loads(e[4])
            assert lab["available_dt"] == tx[lab["transaction_id"]] + 7 * 86_400 == e[0]
    # transaction payloads never carry the label
    first_tx = next(e for e in events if e[2] == s.topic_transactions)
    assert (
        "is_fraud" not in first_tx[4].decode().lower()
        and "isfraud" not in first_tx[4].decode().lower()
    )


def test_label_gate() -> None:
    g = LabelGate()
    assert not g.due(100)
    g.advance(150)
    assert g.due(100)
    g.held[("labels", 0)] = (5, 200)
    assert g.resumable() == []
    g.advance(250)
    assert g.resumable() == [("labels", 0, 5)]


def test_psi_zero_for_same_distribution_and_large_for_shift() -> None:
    rng = np.random.default_rng(0)
    ref_vals = rng.normal(size=20_000)
    ref = build_score_reference(ref_vals)
    assert feature_psi(ref, rng.normal(size=20_000)) < 0.02
    assert feature_psi(ref, rng.normal(1.5, size=20_000)) > 0.25
    assert psi_from_props(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0


def test_reference_covers_every_feature(bundle, synthetic_raw) -> None:  # type: ignore[no-untyped-def]
    enc = bundle.vocab.encode_frame(build_features(synthetic_raw))
    ref = build_reference(enc)
    assert set(ref) == set(enc.columns)


def test_metrics_perfect_and_random_scores() -> None:
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    amt = np.arange(1, 11, dtype=float)
    m = evaluate(y, y.astype(float), amt, budgets=(0.2,))
    assert m["pr_auc"] == 1.0 and m["recall_at_20pct"] == 1.0
    assert m["money_saved_at_20pct"] == 1 + 5


def test_only_labels_known_at_time_t_are_used() -> None:
    day = 86_400
    df = pd.DataFrame(
        {
            "event_dt": [1 * day, 2 * day, 9 * day],
            "available_dt": [8 * day, 9 * day, 16 * day],
            "is_fraud": [1, 0, 1],
            "fraud_score": [0.9, 0.1, 0.8],
        }
    )
    assert len(known_at(df, 8 * day)) == 1
    assert len(known_at(df, 10 * day)) == 2
    series = asof_series(df, start_day=0, end_day=10, step_days=5, window_days=14)
    assert series["labels_known"].tolist() == [0, 2]


def test_repository_inserts_are_idempotent() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    repo = Repository(engine)
    tx = {"transaction_id": 1, "event_dt": 5, "amount": 10.0, "uid": "u", "payload": {"a": 1}}
    sc = {
        "transaction_id": 1,
        "model_name": "m",
        "model_version": "1",
        "fraud_score": 0.2,
        "expected_loss": 2.0,
        "features": {"x": None},
        "latency_ms": 1.0,
    }
    repo.save_scored([tx], [sc])
    repo.save_scored([tx], [sc])
    repo.save_labels([{"transaction_id": 1, "is_fraud": True, "available_dt": 99}] * 2)
    with engine.connect() as c:
        assert c.execute(text("select count(*) from scores")).scalar() == 1
        assert c.execute(text("select count(*) from labels")).scalar() == 1


def test_training_is_reproducible(synthetic_raw) -> None:  # type: ignore[no-untyped-def]
    from fraudrt.features.encoding import Vocab
    from fraudrt.modeling.train import default_params, fit_booster

    feats = build_features(synthetic_raw)
    cut = int(len(feats) * 0.8)
    train, val = feats.iloc[:cut], feats.iloc[cut:]
    vocab = Vocab.fit(train, min_count=1)
    params = {**default_params(5), "num_threads": 2, "num_leaves": 15}
    preds = []
    for _ in range(2):
        b = fit_booster(train, val, vocab, params, max_rounds=30, patience=10)
        preds.append(b.predict(vocab.encode_frame(val)))
    assert np.array_equal(preds[0], preds[1])
