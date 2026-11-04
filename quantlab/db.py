from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from .models import Base

DEFAULT_DB = os.environ.get("QUANTLAB_DB", os.path.expanduser("~/.quantlab/quantlab.db"))

_engine = None
_Session = None


def get_engine(db_path: str | None = None):
    global _engine, _Session
    if _engine is None:
        path = db_path or DEFAULT_DB
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}")

        @event.listens_for(_engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db(db_path: str | None = None):
    Base.metadata.create_all(get_engine(db_path))


@contextmanager
def session(db_path: str | None = None):
    get_engine(db_path)
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def reset_engine_for_tests():
    global _engine, _Session
    _engine = None
    _Session = None
