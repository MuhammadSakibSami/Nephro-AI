"""ORM models for the research dataset.

Currently one table, ``screenings`` — one row per screening API call. The raw
inputs, full prediction, and LLM guidance are stored as JSON (JSONB on
PostgreSQL) so the schema can grow without migrations, while a handful of typed
columns (risk_class, confidence, ...) keep common research queries fast.

To add more entities later (e.g. patients, feedback), add new model classes
here and generate an Alembic migration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

# JSON on every backend; JSONB (indexable, binary) on PostgreSQL.
JSONType = JSON().with_variant(JSONB, "postgresql")


class Screening(Base):
    __tablename__ = "screenings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Which endpoint produced this row: "predict" | "generate-guidance".
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # --- Raw patient input (exactly what was submitted) ---------------------
    input_features: Mapped[Dict[str, Any]] = mapped_column(JSONType, nullable=False)
    n_missing_features: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- ML prediction (typed columns for querying + full JSON) -------------
    risk_class: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    risk_class_display: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    probabilities: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    prediction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)

    # --- LLM guidance (only for /generate-guidance) -------------------------
    guidance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    safety_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    llm_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_spec: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"<Screening id={self.id} source={self.source!r} "
                f"risk={self.risk_class!r} at={self.created_at}>")
