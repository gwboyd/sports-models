import json
import subprocess
import sys
import textwrap
from pathlib import Path

from mangum import Mangum

import main
from src.sports.football import expected_points_api


handler = Mangum(main.app)


def test_dispatch_audit_logs_with_preinstalled_lambda_handler():
    # A fresh interpreter reproduces Lambda installing a WARN-level root handler
    # before main imports; pytest's own logging configuration would hide this bug.
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import io
            import logging
            from types import SimpleNamespace

            output = io.StringIO()
            handler = logging.StreamHandler(output)
            root = logging.getLogger()
            root.handlers = [handler]
            root.setLevel(logging.WARNING)
            import main

            main.run_scheduled_expected_points_update = lambda event: {"status": "success"}
            main.handler(
                {"job": "expected_points_update", "run_key": "api:nfl:audit"},
                SimpleNamespace(aws_request_id="request-123"),
            )
            assert root.handlers == [handler]
            assert "run_key=api:nfl:audit aws_request_id=request-123" in output.getvalue()
        """)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_lambda_handler(monkeypatch):
    monkeypatch.setattr(
        expected_points_api,
        "get_expected_points_picks",
        lambda *_args, **_kwargs: [
            {
                "season": 2026,
                "week": "1",
                "home_team": "A",
                "away_team": "B",
                "home_score_pred": 27.0,
                "away_score_pred": 20.0,
                "spread_pred": -7.0,
                "spread_line": -3.5,
                "spread_play": "A",
                "spread_win_prob": 61.0,
                "spread_lock": 1,
                "total_pred": 47.0,
                "total_line": 44.5,
                "total_play": "over",
                "total_win_prob": 58.0,
                "total_lock": 1,
                "game_id": "1",
                "year_week": "2026_1",
                "date_time": "2026-09-01-17:00",
                "write_time": "2026-08-01 00:00:00",
            }
        ],
    )
    monkeypatch.setattr(main, "API_KEYS", {main.hash_key("test-key"): ["read"]})

    event = {
        "resource": "/nfl-picks",
        "path": "/nfl-picks",
        "httpMethod": "GET",
        "headers": {
            "Authorization": "test-key"
        },
        "queryStringParameters": {},
        "multiValueQueryStringParameters": None,
        "pathParameters": None,
        "stageVariables": None,
        "requestContext": {
            "resourcePath": "/nfl-picks",
            "httpMethod": "GET",
            "path": "/nfl-picks",
            "identity": {
                "sourceIp": "127.0.0.1",
                "userAgent": "Mozilla/5.0"
            }
        },
        "body": None,
        "isBase64Encoded": False
    }

    response = handler(event, {})
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert body[0]["game_id"] == "1"
