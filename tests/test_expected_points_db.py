from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.utils.db import sports_models_db


class FakeCursor:
    def __init__(self):
        self.calls = []
        self.return_time = datetime(2026, 8, 1, tzinfo=timezone.utc)
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.calls.append(("execute", query, params))

    def executemany(self, query, params):
        self.calls.append(("executemany", query, params))

    def fetchone(self):
        return {"id": 42, "write_time": self.return_time}


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def update_record(cursor, *, client_name="pytest"):
    return {
        "year_week": "2026_1",
        "write_time": cursor.return_time,
        "week": 1,
        "season": 2026,
        "environment": "TEST",
        "client_name": client_name,
        "runtime": 1.0,
        "model_version": "1.0",
        "pick_changes": 0,
        "pick_changes_games": [],
        "play_changes": 0,
        "play_changes_games": [],
        "updates_skipped": 0,
        "picks_num": 0,
        "difference_df": [],
        "picks_df": [],
    }


def test_atomic_cfb_run_uses_only_cfb_tables(monkeypatch):
    cursor = FakeCursor()

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    update = update_record(cursor)
    pick_record = {column: 1 for column in sports_models_db.PICK_COLUMNS}
    pick_record.update(
        {
            "week": "1",
            "year_week": "2026_1",
            "game_id": "1",
            "model_version": "1.0",
            "home_conference": "SEC",
            "away_conference": "Big Ten",
        }
    )

    result = sports_models_db.write_expected_points_run(
        ExpectedPointsLeague.CFB,
        [pick_record],
        update,
        write_authorized=True,
    )

    assert result == cursor.return_time
    sql = "\n".join(call[1] for call in cursor.calls)
    assert "cfb_expected_points_pick_updates" in sql
    assert "cfb_expected_points_picks" in sql
    assert "home_conference" in sql
    assert "away_conference" in sql
    assert "nfl_expected_points" not in sql


def test_non_aws_database_write_requires_explicit_authorization(monkeypatch):
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    with pytest.raises(PermissionError, match="explicit authorization"):
        sports_models_db.write_expected_points_run(
            ExpectedPointsLeague.NFL,
            [],
            {},
        )


def test_scheduled_run_links_update_id_and_completion_in_same_transaction(monkeypatch):
    cursor = FakeCursor()
    monkeypatch.setenv(
        "EXPECTED_POINTS_RUN_KEY",
        "aws-scheduler:nfl:2026:1:daily:2026-09-01",
    )

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    sports_models_db.write_expected_points_run(
        ExpectedPointsLeague.NFL,
        [],
        update_record(cursor, client_name="aws-scheduler"),
        write_authorized=True,
    )

    sql = "\n".join(call[1] for call in cursor.calls)
    assert "set update_id = %s" in sql
    assert "status = 'completed'" in sql
    completion_call = next(call for call in cursor.calls if "set update_id = %s" in call[1])
    assert completion_call[2] == (
        42,
        "aws-scheduler:nfl:2026:1:daily:2026-09-01",
        "nfl",
        "nfl_expected_points",
        2026,
        1,
    )


def test_manual_run_does_not_touch_schedule_table(monkeypatch):
    cursor = FakeCursor()
    monkeypatch.delenv("EXPECTED_POINTS_RUN_KEY", raising=False)

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    sports_models_db.write_expected_points_run(
        ExpectedPointsLeague.NFL,
        [],
        update_record(cursor, client_name="notebook"),
        write_authorized=True,
    )

    assert not any("set update_id = %s" in call[1] for call in cursor.calls)


def test_invalid_league_cannot_be_used_as_identifier():
    try:
        sports_models_db.get_expected_points_picks("cfb; drop table x")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid league identifier was accepted")


def test_scheduled_update_claim_is_atomic_and_leased(monkeypatch):
    cursor = FakeCursor()
    cursor.fetchone = lambda: {"status": "running"}

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)

    result = sports_models_db.claim_scheduled_model_update(
        ExpectedPointsLeague.NFL,
        "aws-scheduler:nfl:2026:1:opening",
        2026,
        1,
        lease=timedelta(minutes=16),
    )

    assert result == "claimed"
    _, sql, params = cursor.calls[0]
    assert "status = 'running'" in sql
    assert "attempt_count = attempt_count + 1" in sql
    assert "claimed_at < now() - %s" in sql
    assert params == (
        "aws-scheduler:nfl:2026:1:opening",
        "nfl",
        "nfl_expected_points",
        2026,
        1,
        timedelta(minutes=16),
    )


def test_scheduled_update_claim_reports_existing_terminal_status(monkeypatch):
    cursor = FakeCursor()
    responses = iter([None, {"status": "completed"}])
    cursor.fetchone = lambda: next(responses)

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)

    assert sports_models_db.claim_scheduled_model_update(
        ExpectedPointsLeague.CFB,
        "aws-scheduler:cfb:2026:1:opening",
        2026,
        1,
    ) == "completed"
    assert len(cursor.calls) == 2
