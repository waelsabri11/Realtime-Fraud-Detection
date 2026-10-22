"""`fraudrt` command line: train, bootstrap, replay, worker, allocate, monitor, retrain, serve."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from fraudrt.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def prepare() -> None:
    """Load the Kaggle CSVs and build the offline feature table (cached under work/)."""
    from fraudrt.data import load_features, split_masks

    s = get_settings()
    feats = load_features(s)
    for name, m in split_masks(feats, s).items():
        typer.echo(f"{name}: {int(m.sum())} rows, fraud rate {feats.loc[m, 'isFraud'].mean():.4f}")


@app.command()
def train(promote: bool = True) -> None:
    """Train LightGBM, log to MLflow, register the model (alias `champion` if --promote)."""
    from fraudrt.modeling.train import run_training

    s = get_settings()
    version, metrics = run_training(s, promote=promote)
    out = s.results_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "training.json").write_text(json.dumps({"version": version, **metrics}, indent=2))
    typer.echo(json.dumps({"version": version, **metrics}, indent=2))


@app.command()
def bootstrap() -> None:
    """Warm Redis with the history preceding the replay period."""
    from fraudrt.bootstrap import bootstrap_redis

    typer.echo(bootstrap_redis(get_settings()))


@app.command()
def replay(
    speedup: Annotated[float, typer.Option(help="0 = as fast as possible")] = 0.0,
    limit: Annotated[int | None, typer.Option(help="only the first N transactions")] = None,
    wait: Annotated[bool, typer.Option(help="wait until the worker drained the topics")] = True,
) -> None:
    """Publish the hold-out period (transactions + delayed labels) to Redpanda."""
    import time

    from fraudrt.streaming.kafka_util import group_lag
    from fraudrt.streaming.replayer import replay as do_replay

    s = get_settings()
    t0 = time.perf_counter()
    summary = do_replay(s, speedup=speedup, limit=limit)
    typer.echo(summary)
    if wait:
        while (lag := group_lag(s)) > 0:
            typer.echo(f"consumer lag: {lag}")
            time.sleep(5)
        typer.echo("drained")
        total = time.perf_counter() - t0
        summary["end_to_end_seconds"] = round(total, 1)
        summary["end_to_end_txn_per_second"] = round(summary["transactions"] / total, 1)
        s.results_dir.mkdir(parents=True, exist_ok=True)
        (s.results_dir / "replay.json").write_text(json.dumps(summary, indent=2))


@app.command()
def worker(idle_exit: Annotated[float | None, typer.Option()] = None) -> None:
    """Stream worker: transactions -> /score, labels -> Postgres."""
    from fraudrt.streaming.worker import run_worker

    run_worker(get_settings(), stop_after_idle_seconds=idle_exit)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:
    import uvicorn

    uvicorn.run("fraudrt.api.app:app_factory", factory=True, host=host, port=port, workers=workers)


@app.command()
def reset() -> None:
    """Empty Postgres tables, Redis feature keys and the Kafka topics (fresh replay)."""
    import redis

    from fraudrt.db.repo import Repository, make_engine
    from fraudrt.features.online import RedisFeatureStore
    from fraudrt.streaming.kafka_util import delete_topics

    s = get_settings()
    Repository(make_engine(s.database_url)).truncate_all()
    RedisFeatureStore(redis.Redis.from_url(s.redis_url), s.feature_prefix).flush()
    delete_topics(s)
    typer.echo("reset done")


@app.command()
def purge_loadtest() -> None:
    """Delete what a load test wrote: Postgres rows with ids >= 900,000,000 and their Redis keys."""
    from sqlalchemy import text

    from fraudrt.db.repo import make_engine

    with make_engine(get_settings().database_url).begin() as conn:
        for table in ("scores", "transactions"):
            n = conn.execute(
                text(f"DELETE FROM {table} WHERE transaction_id >= 900000000")
            ).rowcount
            typer.echo(f"{table}: deleted {n}")
    import redis

    r = redis.Redis.from_url(get_settings().redis_url)
    prefix = get_settings().feature_prefix
    keys = []
    for pattern in (f"{prefix}:z:900??????|*", f"{prefix}:last:900??????|*", f"{prefix}:seen:9*"):
        keys += list(r.scan_iter(match=pattern, count=5000))
    for i in range(0, len(keys), 5000):
        r.delete(*keys[i : i + 5000])
    typer.echo(f"redis: deleted {len(keys)} keys")


@app.command()
def allocate(
    capacity: Annotated[int | None, typer.Option()] = None,
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Daily knapsack vs baselines on the replayed scores; persists decisions to Postgres."""
    from fraudrt.reports import run_allocation_report

    s = get_settings()
    typer.echo(
        json.dumps(
            run_allocation_report(s, capacity or s.review_capacity_units, out or s.results_dir),
            indent=2,
        )
    )


@app.command()
def monitor(out: Annotated[Path | None, typer.Option()] = None) -> None:
    """Drift (PSI) and label-aware performance report on the replayed stream."""
    from fraudrt.db.repo import make_engine
    from fraudrt.modeling.registry import load_from_registry
    from fraudrt.monitoring.report import build_report

    s = get_settings()
    bundle = load_from_registry(s)
    report = build_report(s, make_engine(s.database_url), bundle, out or s.results_dir)
    typer.echo(json.dumps(report["drift"]["top_features"][:8], indent=2))


@app.command()
def retrain(as_of_day: int, out: Annotated[Path | None, typer.Option()] = None) -> None:
    """Retrain with only the labels known at `as_of_day`; compare challenger to champion."""
    from fraudrt.db.repo import make_engine
    from fraudrt.modeling.registry import load_from_registry
    from fraudrt.retrain import run_retrain

    s = get_settings()
    res = run_retrain(s, make_engine(s.database_url), load_from_registry(s), as_of_day)
    d = out or s.results_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "retrain.json").write_text(json.dumps(res, indent=2, default=float))
    typer.echo(json.dumps({k: res[k] for k in ("champion", "challenger")}, indent=2, default=float))


if __name__ == "__main__":
    app()
