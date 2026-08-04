from __future__ import annotations

import pytest

from src.sports.football.cfb.expected_points.cfbd_client import CFBDClient


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def close(self):
        self.closed = True


def test_client_uses_expected_endpoint_headers_parameters_and_timeout():
    response = FakeResponse([{"gameId": 1}])
    session = FakeSession(response)

    with CFBDClient("token", session=session) as client:
        assert client.get_lines(2026, "regular") == [{"gameId": 1}]

    assert response.raise_called is True
    assert session.closed is True
    assert session.calls == [
        (
            "https://api.collegefootballdata.com/lines",
            {
                "headers": {"Authorization": "Bearer token", "Accept": "application/json"},
                "params": {"year": 2026, "seasonType": "regular"},
                "timeout": 30,
            },
        )
    ]


@pytest.mark.parametrize(
    ("method_name", "endpoint"),
    [
        ("get_games", "/games"),
        ("get_predicted_points_added_by_game", "/ppa/games"),
        ("get_advanced_game_stats", "/stats/game/advanced"),
    ],
)
def test_client_maps_remaining_endpoints(method_name, endpoint):
    session = FakeSession(FakeResponse([]))
    client = CFBDClient("token", timeout=12, session=session)

    getattr(client, method_name)(2025, "postseason")

    assert session.calls[0][0] == f"https://api.collegefootballdata.com{endpoint}"
    assert session.calls[0][1]["params"] == {"year": 2025, "seasonType": "postseason"}
    assert session.calls[0][1]["timeout"] == 12


def test_client_rejects_non_list_payloads_and_closes_sessions():
    session = FakeSession(FakeResponse({"data": []}))
    client = CFBDClient("token", session=session)

    with pytest.raises(ValueError, match="Expected a JSON list"):
        client.get_games(2026, "regular")

    client.close()
    assert session.closed is True


def test_client_propagates_http_errors_and_requires_an_api_key():
    response = FakeResponse([], error=RuntimeError("bad gateway"))
    client = CFBDClient("token", session=FakeSession(response))

    with pytest.raises(RuntimeError, match="bad gateway"):
        client.get_games(2026, "regular")
    with pytest.raises(ValueError, match="CFBD_API_KEY is required"):
        CFBDClient("")
