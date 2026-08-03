from unittest.mock import MagicMock

import psycopg

from src.utils import postgres


def test_get_connection_retries_transient_operational_error(monkeypatch):
    connection = MagicMock()
    connect = MagicMock(side_effect=[psycopg.OperationalError("temporary DNS failure"), connection])
    sleep = MagicMock()
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://example.invalid/database")
    monkeypatch.setattr(postgres.psycopg, "connect", connect)
    monkeypatch.setattr(postgres.time, "sleep", sleep)

    with postgres.get_connection() as result:
        assert result is connection

    assert connect.call_count == 2
    sleep.assert_called_once_with(0.5)
    connection.commit.assert_called_once()
    connection.close.assert_called_once()


def test_get_connection_raises_after_bounded_retries(monkeypatch):
    connect = MagicMock(side_effect=psycopg.OperationalError("still unavailable"))
    sleep = MagicMock()
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://example.invalid/database")
    monkeypatch.setattr(postgres.psycopg, "connect", connect)
    monkeypatch.setattr(postgres.time, "sleep", sleep)

    try:
        with postgres.get_connection():
            pass
    except psycopg.OperationalError:
        pass
    else:
        raise AssertionError("Expected the final connection failure to be raised")

    assert connect.call_count == postgres.CONNECTION_ATTEMPTS
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]
