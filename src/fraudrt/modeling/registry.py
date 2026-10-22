"""Model bundle (booster + vocabulary + spec) and loading from the MLflow registry or a directory."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from fraudrt.config import Settings
from fraudrt.features.definitions import FEATURE_COLUMNS
from fraudrt.features.encoding import Vocab

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    booster: lgb.Booster
    vocab: Vocab
    name: str
    version: str
    run_id: str | None = None
    reference: dict[str, Any] | None = None

    def predict(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(self.booster.predict(x, num_threads=1))


def save_local(path: Path, booster: lgb.Booster, vocab: Vocab, spec: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path / "model.txt"))
    (path / "spec.json").write_text(
        json.dumps({**spec, "feature_columns": FEATURE_COLUMNS, "vocab": vocab.to_json()})
    )


def load_local(path: Path, name: str = "local", version: str = "0") -> ModelBundle:
    spec = json.loads((path / "spec.json").read_text())
    if spec["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("model was trained with a different feature set than this code")
    return ModelBundle(
        booster=lgb.Booster(model_file=str(path / "model.txt")),
        vocab=Vocab(spec["vocab"]),
        name=name,
        version=version,
    )


def load_from_registry(settings: Settings) -> ModelBundle:
    """Resolve `models:/<name>@<alias>` or `models:/<name>/<version>` and load booster + spec."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    ref = settings.model_version
    mv = (
        client.get_model_version(settings.model_name, ref)
        if ref.isdigit()
        else client.get_model_version_by_alias(settings.model_name, ref)
    )
    model_dir = Path(mlflow.artifacts.download_artifacts(f"models:/{mv.name}/{mv.version}"))
    booster_file = next(model_dir.glob("model.*"))
    spec_dir = Path(mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/spec"))
    spec = json.loads((spec_dir / "spec.json").read_text())
    if spec["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("model was trained with a different feature set than this code")
    ref_file = spec_dir / "reference.json"
    log.info("loaded %s v%s (run %s)", mv.name, mv.version, mv.run_id)
    return ModelBundle(
        booster=lgb.Booster(model_file=str(booster_file)),
        vocab=Vocab(spec["vocab"]),
        name=mv.name,
        version=str(mv.version),
        run_id=mv.run_id,
        reference=json.loads(ref_file.read_text()) if ref_file.exists() else None,
    )
