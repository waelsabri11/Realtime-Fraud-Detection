"""FastAPI service: POST /score, GET /health."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import redis
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from fraudrt.api.schemas import HealthResponse, ModelInfo, ScoreRequest, ScoreResponse
from fraudrt.api.scorer import Scorer
from fraudrt.api.writer import BatchWriter, NullSink, Sink
from fraudrt.config import Settings, get_settings
from fraudrt.db.repo import Repository, make_engine
from fraudrt.features.online import RedisFeatureStore
from fraudrt.modeling.registry import ModelBundle, load_from_registry

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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = redis_client or redis.Redis.from_url(settings.redis_url)
        the_repo = repo
        if the_repo is None and sink is None:
            the_repo = Repository(make_engine(settings.database_url))
        the_sink: Sink = sink or (BatchWriter(the_repo) if the_repo else NullSink())
        holder = ModelHolder(settings, bundle)
        holder.start_background_load()
        app.state.scorer = Scorer(RedisFeatureStore(client, settings.feature_prefix))
        app.state.holder = holder
        app.state.redis = client
        app.state.repo = the_repo
        app.state.sink = the_sink
        yield
        holder.stop()
        the_sink.close()

    app = FastAPI(
        title="Real-time fraud scoring (replay-simulated)", version="0.1.0", lifespan=lifespan
    )

    def get_holder(request: Request) -> ModelHolder:
        holder: ModelHolder = request.app.state.holder
        return holder

    @app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
    def score(
        req: ScoreRequest,
        request: Request,
        holder: Annotated[ModelHolder, Depends(get_holder)],
        include_features: Annotated[bool, Query()] = False,
    ) -> ScoreResponse:
        t0 = time.perf_counter()
        model = holder.bundle
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        scored = request.app.state.scorer.score(req, model)
        latency_ms = (time.perf_counter() - t0) * 1e3
        request.app.state.sink.submit(scored, latency_ms)
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

    @app.post("/admin/reload")
    def reload_model(holder: Annotated[ModelHolder, Depends(get_holder)]) -> dict[str, Any]:
        ok = holder.try_load()
        b = holder.bundle
        return {"reloaded": ok, "version": b.version if b else None}

    return app


def _features_out(feats: dict[str, Any]) -> dict[str, float | str | None]:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in feats.items()}


def app_factory() -> FastAPI:  # `uvicorn fraudrt.api.app:app_factory --factory`
    return create_app()
