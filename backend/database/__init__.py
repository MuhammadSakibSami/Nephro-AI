"""Persistence layer for the kidney-risk screening API.

A thin SQLAlchemy 2.0 layer over PostgreSQL that stores every screening
(patient inputs + ML prediction + LLM guidance) for the research dataset.

Design goals:
  * **Graceful degradation** — if ``DATABASE_URL`` is unset or the database is
    unreachable, the API keeps working; saving is skipped and logged. Screening
    is never blocked by the database.
  * **Forward-compatible** — inputs/prediction/guidance are stored as JSON
    (JSONB on PostgreSQL) plus a few typed columns for easy querying, so new
    fields can be added later without a migration.

Public surface:
    from .database import init_db, db_status, save_screening
"""
from __future__ import annotations

from .db import init_db, db_status, is_available, session_scope, Base
from .models import Screening
from .repository import save_screening, count_screenings

__all__ = [
    "init_db",
    "db_status",
    "is_available",
    "session_scope",
    "Base",
    "Screening",
    "save_screening",
    "count_screenings",
]
