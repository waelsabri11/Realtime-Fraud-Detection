"""Off-request-path persistence: scored transactions are batched into Postgres by a thread."""

from __future__ import annotations

import logging
import math
import queue
import threading
from datetime import UTC, datetime
from typing import Any, Protocol

from fraudrt.api.scorer import ScoredTransaction
from fraudrt.db.repo import Repository

log = logging.getLogger(__name__)


class Sink(Protocol):
    def submit(self, item: ScoredTransaction, latency_ms: float) -> None: ...
    def close(self) -> None: ...


def _json_safe(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


class NullSink:
    def submit(self, item: ScoredTransaction, latency_ms: float) -> None:
        return None

    def close(self) -> None:
        return None


class BatchWriter:
    def __init__(
        self, repo: Repository, flush_interval: float = 0.25, max_batch: int = 500
    ) -> None:
        self.repo = repo
        self.flush_interval = flush_interval
        self.max_batch = max_batch
        self.q: queue.Queue[tuple[dict[str, Any], dict[str, Any]]] = queue.Queue(maxsize=100_000)
        self.dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()

    def submit(self, item: ScoredTransaction, latency_ms: float) -> None:
        if item.duplicate:
            return
        req = item.request
        now = datetime.now(UTC)  # the moment the score existed, not when the batch is flushed
        tx = {
            "transaction_id": req.transaction_id,
            "event_dt": req.transaction_dt,
            "amount": req.transaction_amt,
            "uid": item.uid,
            "payload": _json_safe(req.to_row()),
            "received_at": now,
        }
        sc = {
            "transaction_id": req.transaction_id,
            "model_name": item.bundle.name,
            "model_version": item.bundle.version,
            "fraud_score": item.score,
            "expected_loss": item.expected_loss,
            "features": _json_safe(item.features),
            "latency_ms": latency_ms,
            "scored_at": now,
        }
        try:
            self.q.put_nowait((tx, sc))
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> None:
        txs: list[dict[str, Any]] = []
        scs: list[dict[str, Any]] = []
        while len(txs) < self.max_batch:
            try:
                tx, sc = self.q.get_nowait()
            except queue.Empty:
                break
            txs.append(tx)
            scs.append(sc)
        if txs:
            try:
                self.repo.save_scored(txs, scs)
            except Exception:
                log.exception("failed to persist %d scored transactions", len(txs))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.flush_interval)
            while not self.q.empty():
                self._drain()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        while not self.q.empty():
            self._drain()
