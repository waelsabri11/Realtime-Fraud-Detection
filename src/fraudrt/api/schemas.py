"""Pydantic request / response schemas for the scoring API."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudrt.features.definitions import RAW_FIELDS

# explicit request field -> raw IEEE-CIS column
_EXPLICIT: dict[str, str] = {
    "transaction_id": "TransactionID",
    "transaction_dt": "TransactionDT",
    "transaction_amt": "TransactionAmt",
    "product_cd": "ProductCD",
    "card1": "card1",
    "card2": "card2",
    "card3": "card3",
    "card4": "card4",
    "card5": "card5",
    "card6": "card6",
    "addr1": "addr1",
    "addr2": "addr2",
    "dist1": "dist1",
    "dist2": "dist2",
    "p_emaildomain": "P_emaildomain",
    "r_emaildomain": "R_emaildomain",
}
ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(RAW_FIELDS) - frozenset(_EXPLICIT.values())

AttrValue = float | str | None


class ScoreRequest(BaseModel):
    """One transaction as the payment platform would send it (no label)."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int = Field(ge=0)
    transaction_dt: int = Field(ge=0, description="Seconds on the (replay) clock")
    transaction_amt: float = Field(gt=0)
    product_cd: str | None = None
    card1: int | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    p_emaildomain: str | None = None
    r_emaildomain: str | None = None
    attributes: dict[str, AttrValue] = Field(
        default_factory=dict, description="Other IEEE-CIS columns: C*, D*, M*, id_*, DeviceType"
    )

    @field_validator("attributes")
    @classmethod
    def _known_attributes(cls, v: dict[str, AttrValue]) -> dict[str, AttrValue]:
        unknown = sorted(set(v) - ALLOWED_ATTRIBUTES)
        if unknown:
            raise ValueError(f"unknown attributes: {unknown}")
        return v

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {raw: getattr(self, f) for f, raw in _EXPLICIT.items()}
        row.update(self.attributes)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ScoreRequest:
        def clean(v: Any) -> Any:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return v.item() if hasattr(v, "item") else v

        kwargs: dict[str, Any] = {f: clean(row.get(raw)) for f, raw in _EXPLICIT.items()}
        for f in ("transaction_id", "transaction_dt", "card1"):
            if kwargs[f] is not None:
                kwargs[f] = int(kwargs[f])
        kwargs["attributes"] = {
            c: clean(row[c]) for c in ALLOWED_ATTRIBUTES if c in row and clean(row[c]) is not None
        }
        return cls(**kwargs)


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_score: float = Field(ge=0, le=1)
    expected_loss: float = Field(
        description="fraud_score x amount: the value of reviewing this case"
    )
    model_name: str
    model_version: str
    latency_ms: float
    duplicate: bool = False
    features: dict[str, float | str | None] | None = None


class ModelInfo(BaseModel):
    name: str
    version: str


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo | None
    redis: bool
    database: bool
