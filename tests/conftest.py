from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fakeredis
import pytest

from fraudrt.config import Settings
from fraudrt.data import build_features
from fraudrt.features.encoding import Vocab
from fraudrt.modeling.registry import ModelBundle
from fraudrt.modeling.train import default_params, fit_booster
from fraudrt.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_raw():  # type: ignore[no-untyped-def]
    return make_synthetic(n=6000, n_cards=200, days=40, seed=3)


@pytest.fixture(scope="session")
def bundle(synthetic_raw):  # type: ignore[no-untyped-def]
    feats = build_features(synthetic_raw)
    cut = int(len(feats) * 0.8)
    train, val = feats.iloc[:cut], feats.iloc[cut:]
    vocab = Vocab.fit(train, min_count=1)
    params = {**default_params(0), "num_threads": 2, "learning_rate": 0.1, "num_leaves": 15}
    booster = fit_booster(train, val, vocab, params, max_rounds=60, patience=10)
    return ModelBundle(booster=booster, vocab=vocab, name="fraud-test", version="1")


@pytest.fixture
def fake_redis() -> Iterator[fakeredis.FakeRedis]:
    yield fakeredis.FakeRedis()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(work_dir=tmp_path, _env_file=None)
