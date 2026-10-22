"""Offline (vectorised) vs online (Redis, sequential) feature parity."""

from __future__ import annotations

import fakeredis
import numpy as np
import pandas as pd

from fraudrt.data import build_features
from fraudrt.features.behavioral import add_uid, behavioral_from_events
from fraudrt.features.definitions import BEHAVIORAL, FEATURE_COLUMNS, make_uid, stateless_features
from fraudrt.features.encoding import Vocab
from fraudrt.features.online import RedisFeatureStore
from fraudrt.synthetic import make_synthetic


def _row_dicts(df: pd.DataFrame) -> list[dict[str, object]]:
    return df.astype(object).where(df.notna(), None).to_dict("records")  # type: ignore[no-any-return]


def test_online_matches_offline_behavioral() -> None:
    raw = make_synthetic(n=3000, n_cards=120, days=30, seed=1)
    off = build_features(raw)
    store = RedisFeatureStore(fakeredis.FakeRedis())
    rows = _row_dicts(raw.sort_values(["TransactionDT", "TransactionID"]))
    online = []
    for r in rows:
        uid = make_uid(r)
        ts = int(r["TransactionDT"])  # type: ignore[call-overload]
        amt = float(r["TransactionAmt"])  # type: ignore[arg-type]
        online.append(store.get_features(uid, ts, amt))
        store.update(uid, int(r["TransactionID"]), ts, round(amt * 100))  # type: ignore[call-overload]
    on = pd.DataFrame(online)[BEHAVIORAL]
    off = off.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    np.testing.assert_allclose(
        off[BEHAVIORAL].to_numpy(), on.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )
    assert off["uid_cnt_7d"].max() > 3  # windows really exercised


def test_tied_timestamps_and_first_seen() -> None:
    now = 1000
    f = behavioral_from_events(now, 10.0, None, [])
    assert f["uid_cnt_7d"] == 0 and np.isnan(f["uid_secs_since_last"])
    f = behavioral_from_events(now, 10.0, 1000, [(1000, 500), (now - 3600, 1500)])
    assert f["uid_cnt_1h"] == 2 and f["uid_secs_since_last"] == 0
    assert f["uid_mean_7d"] == 10.0 and f["uid_amt_ratio_7d"] == 1.0


def test_stateless_features_match_offline_frame_and_encoding() -> None:
    raw = make_synthetic(n=500, seed=2)
    off = build_features(raw)
    vocab = Vocab.fit(off, min_count=1)
    enc = vocab.encode_frame(off)
    for i in (0, 17, 250, 499):
        r = _row_dicts(raw)[i]
        feats = stateless_features(r)
        feats.update(off.loc[i, BEHAVIORAL].to_dict())
        vec = vocab.encode_row(feats)
        np.testing.assert_allclose(vec, enc.iloc[i][FEATURE_COLUMNS].to_numpy(), equal_nan=True)


def test_uid_normalises_numeric_representation() -> None:
    a = make_uid({"card1": 13926, "card2": None, "card4": "visa", "addr1": 315.0})
    b = make_uid({"card1": 13926.0, "card2": float("nan"), "card4": "visa", "addr1": 315})
    assert a == b
    assert len(add_uid(make_synthetic(n=50))) == 50
