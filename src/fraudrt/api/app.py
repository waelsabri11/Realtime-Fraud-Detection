"""FastAPI service: POST /score, GET /health."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import redis
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraudrt.api.schemas import HealthResponse, ModelInfo, ScoreRequest, ScoreResponse
from fraudrt.api.scorer import Scorer
from fraudrt.api.writer import BatchWriter, NullSink, Sink
from fraudrt.config import Settings, get_settings
from fraudrt.db.repo import Repository, make_engine
from fraudrt.features.online import RedisFeatureStore
from fraudrt.modeling.registry import ModelBundle, load_from_registry
from fraudrt.observability import ApiMetrics, DriftRefresher, MetricsMiddleware

log = logging.getLogger("fraudrt.api")


class ModelHolder:
    """Holds the active model; keeps trying to load it until the registry has one."""

    def __init__(self, settings: Settings, bundle: ModelBundle | None = None) -> None:
        self.settings = settings
        self.bundle = bundle
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def try_load(self) -> bool:
        try:
            self.bundle = load_from_registry(self.settings)
            return True
        except Exception as exc:
            log.warning("model not loadable yet (%s: %s)", type(exc).__name__, exc)
            return False

    def start_background_load(self) -> None:
        if self.bundle is not None:
            return

        def loop() -> None:
            while not self._stop.is_set() and self.bundle is None:
                if self.try_load():
                    return
                self._stop.wait(self.settings.model_reload_seconds)

        self._thread = threading.Thread(target=loop, name="model-loader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def create_app(
    settings: Settings | None = None,
    *,
    bundle: ModelBundle | None = None,
    redis_client: redis.Redis | None = None,
    repo: Repository | None = None,
    sink: Sink | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    metrics = ApiMetrics(
        settings.alert_score_threshold, settings.drift_window, settings.drift_min_samples
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = redis_client or redis.Redis.from_url(settings.redis_url)
        the_repo = repo
        if the_repo is None and sink is None:
            the_repo = Repository(make_engine(settings.database_url))
        the_sink: Sink = sink or (BatchWriter(the_repo) if the_repo else NullSink())
        holder = ModelHolder(settings, bundle)
        holder.start_background_load()
        refresher = DriftRefresher(
            metrics,
            lambda: holder.bundle.reference if holder.bundle else None,
            settings.drift_interval_seconds,
        )
        refresher.start()
        app.state.scorer = Scorer(RedisFeatureStore(client, settings.feature_prefix))
        app.state.holder = holder
        app.state.redis = client
        app.state.repo = the_repo
        app.state.sink = the_sink
        yield
        refresher.stop()
        holder.stop()
        the_sink.close()

    app = FastAPI(
        title="Real-time fraud scoring (replay-simulated)", version="0.1.0", lifespan=lifespan
    )
    app.add_middleware(MetricsMiddleware, metrics=metrics)
    app.state.metrics = metrics

    @app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
    def score(
        req: ScoreRequest,
        request: Request,
        include_features: Annotated[bool, Query()] = False,
    ) -> ScoreResponse:
        t0 = time.perf_counter()
        holder: ModelHolder = request.app.state.holder
        model = holder.bundle
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        scored = request.app.state.scorer.score(req, model)
        latency_ms = (time.perf_counter() - t0) * 1e3
        request.app.state.sink.submit(scored, latency_ms)
        if scored.encoded is not None:
            metrics.observe_scored(
                scored.score, scored.encoded, scored.duplicate, model.name, model.version
            )
        log.info(
            "score txn=%d p=%.4f dup=%s latency_ms=%.2f features_ms=%.2f redis_ms=%.2f model_ms=%.2f",
            req.transaction_id,
            scored.score,
            scored.duplicate,
            latency_ms,
            scored.timings_ms["features"],
            scored.timings_ms["redis_read"] + scored.timings_ms["redis_write"],
            scored.timings_ms["model"],
        )
        return ScoreResponse(
            transaction_id=req.transaction_id,
            fraud_score=scored.score,
            expected_loss=scored.expected_loss,
            model_name=model.name,
            model_version=model.version,
            latency_ms=latency_ms,
            duplicate=scored.duplicate,
            features=_features_out(scored.features) if include_features else None,
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        holder: ModelHolder = request.app.state.holder
        try:
            redis_ok = bool(request.app.state.redis.ping())
        except Exception:
            redis_ok = False
        db_ok = True
        if request.app.state.repo is not None:
            try:
                db_ok = request.app.state.repo.ping()
            except Exception:
                db_ok = False
        m = holder.bundle
        ok = m is not None and redis_ok and db_ok
        return HealthResponse(
            status="ok" if ok else "degraded",
            model=ModelInfo(name=m.name, version=m.version) if m else None,
            redis=redis_ok,
            database=db_ok,
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reload")
    def reload_model(request: Request) -> dict[str, Any]:
        holder: ModelHolder = request.app.state.holder
        ok = holder.try_load()
        b = holder.bundle
        return {"reloaded": ok, "version": b.version if b else None}

    return app


def _features_out(feats: dict[str, Any]) -> dict[str, float | str | None]:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in feats.items()}


def app_factory() -> FastAPI:  # `uvicorn fraudrt.api.app:app_factory --factory`
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return create_app()
