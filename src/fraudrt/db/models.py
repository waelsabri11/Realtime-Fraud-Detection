"""Postgres schema: transactions, scores, decisions, labels."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_dt: Mapped[int] = mapped_column(Integer, index=True)  # seconds on the replay clock
    amount: Mapped[float] = mapped_column(Numeric(14, 2))
    uid: Mapped[str] = mapped_column(String)
    payload: Mapped[dict[str, Any]] = mapped_column(Json)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Score(Base):
    __tablename__ = "scores"
    transaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_name: Mapped[str] = mapped_column(String)
    model_version: Mapped[str] = mapped_column(String)
    fraud_score: Mapped[float] = mapped_column(Float)
    expected_loss: Mapped[float] = mapped_column(Float)
    features: Mapped[dict[str, Any]] = mapped_column(Json)
    latency_ms: Mapped[float] = mapped_column(Float)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Label(Base):
    """Ground truth that becomes known `available_dt` seconds on the replay clock."""

    __tablename__ = "labels"
    transaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    is_fraud: Mapped[bool] = mapped_column(Boolean)
    available_dt: Mapped[int] = mapped_column(Integer, index=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Decision(Base):
    """Daily alert-allocation outcome per policy (which transactions go to an analyst)."""

    __tablename__ = "decisions"
    __table_args__ = (
        UniqueConstraint("transaction_id", "policy", name="uq_decision_txn_policy"),
        Index("ix_decisions_policy_day", "policy", "decision_day"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[int] = mapped_column(BigInteger)
    policy: Mapped[str] = mapped_column(String)
    decision_day: Mapped[int] = mapped_column(Integer)
    review_cost: Mapped[int] = mapped_column(Integer)
    expected_loss_saved: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
