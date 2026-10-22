"""initial schema: transactions, scores, labels, decisions

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "transactions",
        sa.Column("transaction_id", sa.BigInteger, primary_key=True),
        sa.Column("event_dt", sa.Integer, nullable=False),
        sa.Column("amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("uid", sa.String, nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=NOW),
    )
    op.create_index("ix_transactions_event_dt", "transactions", ["event_dt"])
    op.create_table(
        "scores",
        sa.Column("transaction_id", sa.BigInteger, primary_key=True),
        sa.Column("model_name", sa.String, nullable=False),
        sa.Column("model_version", sa.String, nullable=False),
        sa.Column("fraud_score", sa.Float, nullable=False),
        sa.Column("expected_loss", sa.Float, nullable=False),
        sa.Column("features", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), server_default=NOW),
    )
    op.create_table(
        "labels",
        sa.Column("transaction_id", sa.BigInteger, primary_key=True),
        sa.Column("is_fraud", sa.Boolean, nullable=False),
        sa.Column("available_dt", sa.Integer, nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=NOW),
    )
    op.create_index("ix_labels_available_dt", "labels", ["available_dt"])
    op.create_table(
        "decisions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("transaction_id", sa.BigInteger, nullable=False),
        sa.Column("policy", sa.String, nullable=False),
        sa.Column("decision_day", sa.Integer, nullable=False),
        sa.Column("review_cost", sa.Integer, nullable=False),
        sa.Column("expected_loss_saved", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW),
        sa.UniqueConstraint("transaction_id", "policy", name="uq_decision_txn_policy"),
    )
    op.create_index("ix_decisions_policy_day", "decisions", ["policy", "decision_day"])


def downgrade() -> None:
    for t in ("decisions", "labels", "scores", "transactions"):
        op.drop_table(t)
