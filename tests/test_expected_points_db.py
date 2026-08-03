from contextlib import contextmanager
from datetime import datetime, timezone

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.utils.db import sports_models_db


class FakeCursor:
    def __init__(self):
        self.calls = []
        self.return_time = datetime(2026, 8, 1, tzinfo=timezone.utc)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.calls.append(("execute", query, params))

    def executemany(self, query, params):
        self.calls.append(("executemany", query, params))

    def fetchone(self):
        return {"write_time": self.return_time}


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_atomic_cfb_run_uses_only_cfb_tables(monkeypatch):
    cursor = FakeCursor()

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    update = {
        "year_week": "2026_1",
        "write_time": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "week": 1,
        "season": 2026,
        "environment": "TEST",
        "client_name": "pytest",
        "runtime": 1.0,
        "pick_changes": 1,
        "pick_changes_games": ["1"],
        "play_changes": 1,
        "play_changes_games": ["1"],
        "updates_skipped": 0,
        "picks_num": 1,
        "difference_df": [],
        "picks_df": [],
    }
    pick_record = {column: 1 for column in sports_models_db.PICK_COLUMNS}
    pick_record.update({"week": "1", "year_week": "2026_1", "game_id": "1"})

    result = sports_models_db.write_expected_points_run(ExpectedPointsLeague.CFB, [pick_record], update)

    assert result == cursor.return_time
    sql = "\n".join(call[1] for call in cursor.calls)
    assert "cfb_expected_points_pick_updates" in sql
    assert "cfb_expected_points_picks" in sql
    assert "nfl_expected_points" not in sql


def test_invalid_league_cannot_be_used_as_identifier():
    try:
        sports_models_db.get_expected_points_picks("cfb; drop table x")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid league identifier was accepted")
