"""Write/read helpers. Saving NEVER raises into the request path.

``save_screening`` is best-effort: on any DB error it logs a warning and returns
None, so a screening request always succeeds even if persistence fails.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from . import db
from .models import Screening

logger = logging.getLogger("ckd.database")


def _count_missing(input_features: Optional[Dict[str, Any]]) -> Optional[int]:
    """How many submitted fields were left blank (None)."""
    if not isinstance(input_features, dict):
        return None
    return sum(1 for v in input_features.values() if v is None)


def save_screening(
    source: str,
    input_features: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
    guidance_response: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Persist one screening. Returns the new row id, or None if skipped/failed.

    * ``source``             "predict" | "generate-guidance"
    * ``input_features``     the raw submitted feature dict
    * ``prediction``         a /predict response dict (typed columns extracted)
    * ``guidance_response``  a build_guidance_response() dict (guidance + safety)
    """
    if not db.is_available():
        return None

    prediction = prediction or {}
    guidance_response = guidance_response or {}

    try:
        row = Screening(
            source=source,
            input_features=input_features,
            n_missing_features=_count_missing(input_features),
            risk_class=prediction.get("risk_class"),
            risk_class_display=prediction.get("risk_class_display"),
            confidence=prediction.get("confidence"),
            probabilities=prediction.get("probabilities"),
            prediction=prediction or None,
            guidance=guidance_response.get("guidance"),
            safety_status=guidance_response.get("status"),
            llm_mode=guidance_response.get("mode"),
            model_spec=guidance_response.get("model"),
        )
        with db.session_scope() as session:
            if session is None:
                return None
            session.add(row)
            session.flush()  # populate row.id before commit
            new_id = row.id
        return new_id
    except Exception as exc:  # never break the request because of persistence
        logger.warning("save_screening failed (continuing without persistence): %s", exc)
        return None


def count_screenings() -> Optional[int]:
    """Total stored screenings, or None if the DB is unavailable."""
    if not db.is_available():
        return None
    try:
        from sqlalchemy import func, select

        with db.session_scope() as session:
            if session is None:
                return None
            return session.execute(select(func.count(Screening.id))).scalar_one()
    except Exception as exc:
        logger.warning("count_screenings failed: %s", exc)
        return None
