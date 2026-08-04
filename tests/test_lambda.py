import json

from mangum import Mangum

import main
from src.sports.football import expected_points_api


handler = Mangum(main.app)


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
