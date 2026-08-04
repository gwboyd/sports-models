from src.sports.basketball.nba.first_basket_model import handler as nba_handler
from src.sports.basketball.nba.first_basket_model.data_model import NBAFirstBasketPick
from src.utils.data_models.picks_results_response import GameResult


def test_nullable_game_result_fields_remain_null_when_omitted():
    game = GameResult(
        season=2026,
        week="1",
        home_team="A",
        away_team="B",
        home_score=0,
        away_score=0,
        home_score_pred=21.0,
        away_score_pred=20.0,
        spread_pred=-1.0,
        spread_line=-1.5,
        true_spread=0.0,
        spread_play="A",
        spread_win_prob=50.0,
        spread_lock=0,
        total_pred=41.0,
        total_line=42.0,
        true_total=0.0,
        total_play="under",
        total_win_prob=50.0,
        total_lock=0,
        year_week="2026_1",
        game_id="game",
        date_time="2026-09-01T17:00:00",
    )

    serialized = game.model_dump()
    assert serialized["correct_spread_play"] is None
    assert serialized["spread_win"] is None
    assert serialized["correct_total_play"] is None
    assert serialized["total_win"] is None


def test_nba_upload_serializes_models_with_model_dump(monkeypatch):
    captured = []
    monkeypatch.setattr(
        nba_handler,
        "replace_nba_first_basket_picks",
        lambda rows: captured.extend(rows) or len(rows),
    )
    pick = NBAFirstBasketPick(
        date="2026-01-01",
        player_name="Player",
        team="TEAM",
        fb_model_prob=0.5,
        fb_model_odds=100.0,
        odds=110.0,
        sportsbook="Book",
        units=1.0,
    )

    assert nba_handler.nba_first_basket_upload([pick]) == {
        "message": "Data uploaded successfully",
        "row_count": 1,
    }
    assert captured == [pick.model_dump()]
