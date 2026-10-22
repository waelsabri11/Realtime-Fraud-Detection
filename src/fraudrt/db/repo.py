"""Persistence helpers (batched inserts are idempotent: ON CONFLICT DO NOTHING)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fraudrt.db.models import Decision, Label, Score, Transaction

log = logging.getLogger(__name__)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


class Repository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def _insert(self, table: Any, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(pg_insert(table).on_conflict_do_nothing(), list(rows))
            else:  # sqlite in unit tests
                conn.execute(table.__table__.insert().prefix_with("OR IGNORE"), list(rows))

    def save_scored(
        self, transactions: Sequence[dict[str, Any]], scores: Sequence[dict[str, Any]]
    ) -> None:
        self._insert(Transaction, transactions)
        self._insert(Score, scores)

    def save_labels(self, rows: Sequence[dict[str, Any]]) -> None:
        self._insert(Label, rows)

    def save_decisions(self, rows: Sequence[dict[str, Any]]) -> None:
        self._insert(Decision, rows)

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def truncate_all(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("TRUNCATE decisions, labels, scores, transactions RESTART IDENTITY"))
