import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row


DEFAULT_SCHEMA = "sports_models"


def get_db_url() -> str:
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("Missing SUPABASE_DB_URL environment variable")
    return db_url


def get_schema() -> str:
    return os.getenv("SUPABASE_SCHEMA", DEFAULT_SCHEMA)


@contextmanager
def get_connection():
    conn = psycopg.connect(
        get_db_url(),
        row_factory=dict_row,
        options=f"-c search_path={get_schema()},public",
        prepare_threshold=None,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        value = value.item()

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    return value


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_scalar(value) for key, value in record.items()}


def json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _json_default(value: Any):
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def normalize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_record(record) for record in records]
