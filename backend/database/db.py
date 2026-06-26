"""Engine / session management with graceful degradation.

The engine is created lazily from ``DATABASE_URL``. If that variable is missing
or the connection test fails, the module stays in a "disabled" state: every
helper becomes a safe no-op and ``is_available()`` returns ``False``. Callers
must treat persistence as best-effort.

Environment variables:
  * ``DATABASE_URL``        SQLAlchemy URL, e.g.
                            postgresql+psycopg2://user:pass@host:5432/ckd_research
                            (SQLite also works for local testing, e.g.
                            sqlite:///./ckd_local.db)
  * ``CKD_DB_AUTO_CREATE``  "true" (default) -> create tables on startup via
                            ``Base.metadata.create_all`` for convenience.
                            Set "false" when you manage schema with Alembic.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger("ckd.database")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# Module-level state (set by init_db()).
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None
_init_error: Optional[str] = None
_initialized: bool = False


def _truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def init_db() -> bool:
    """Create the engine from DATABASE_URL and verify connectivity.

    Returns True if the database is available, False otherwise. Safe to call
    once at startup; never raises.
    """
    global _engine, _SessionLocal, _init_error, _initialized
    _initialized = True

    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        _init_error = "DATABASE_URL is not set; persistence is disabled."
        logger.warning("Database disabled: %s", _init_error)
        return False

    # Hide credentials when logging the target.
    safe_target = url.split("@")[-1] if "@" in url else url

    try:
        engine = create_engine(url, pool_pre_ping=True, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # connection / driver failure -> degrade gracefully
        _engine = None
        _SessionLocal = None
        _init_error = f"could not connect to database ({safe_target}): {exc}"
        logger.warning("Database disabled: %s", _init_error)
        return False

    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    _init_error = None
    logger.info("Database connected: %s", safe_target)

    if _truthy(os.getenv("CKD_DB_AUTO_CREATE"), default=True):
        try:
            # Import models so they register on Base.metadata before create_all.
            from . import models  # noqa: F401

            Base.metadata.create_all(engine)
            logger.info("Database schema ensured (auto-create).")
        except Exception as exc:
            logger.warning("Auto-create tables failed (continuing): %s", exc)

    return True


def is_available() -> bool:
    """True if a working engine/session factory is configured."""
    return _SessionLocal is not None


def get_engine() -> Optional[Engine]:
    return _engine


@contextmanager
def session_scope() -> Iterator[Optional[Session]]:
    """Transactional session context.

    Yields a Session if the DB is available, else yields None (callers should
    skip). Commits on success, rolls back on error, always closes.
    """
    if _SessionLocal is None:
        yield None
        return

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db_status() -> dict:
    """Lightweight status for the /db/health endpoint."""
    return {
        "initialized": _initialized,
        "available": is_available(),
        "auto_create": _truthy(os.getenv("CKD_DB_AUTO_CREATE"), default=True),
        "configured": bool(os.getenv("DATABASE_URL", "").strip()),
        "error": _init_error,
    }
