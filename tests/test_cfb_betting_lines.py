from __future__ import annotations

import pandas as pd
import pytest

from src.model_patterns.expected_points.betting import determine_plays
from src.model_patterns.expected_points.types import PlayThresholds
from src.sports.football.cfb.expected_points.betting_lines import (
    OPTIONAL_CONFIDENCE_FEATURE_GROUPS,
    assemble_cfb_betting_markets,
    scores_to_cfb_bets,
)


def quote(provider, spread, total, *, spread_open=None, total_open=None, home_ml=None, away_ml=None):
    return {
        "provider": provider,
        "spread": spread,
        "overUnder": total,
        "spreadOpen": spread_open,
        "overUnderOpen": total_open,
        "homeMoneyline": home_ml,
        "awayMoneyline": away_ml,
    }


def game(lines, *, game_id=1, home_classification="fbs", away_classification="fbs"):
    return {
        "id": game_id,
        "home_classification": home_classification,
        "away_classification": away_classification,
        "lines": lines,
    }


def with_predictions(markets, *, home=27.0, away=21.0):
    return markets.assign(
        home_team="Clemson",
        away_team="LSU",
        home_score_pred=home,
        away_score_pred=away,
    )


def test_consensus_direction_then_shops_best_corroborated_lines():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("Draft Kings", -2.0, 49.0, spread_open=-1.5, total_open=48.0),
                        quote("ESPN Bet", -3.0, 50.0, spread_open=-2.5, total_open=48.5),
                        quote("Bovada", -3.5, 51.0, spread_open=-3.0, total_open=49.0),
                    ]
                )
            ]
        )
    )

    bets = scores_to_cfb_bets(with_predictions(markets, home=27.0, away=21.0))
    row = bets.iloc[0]

    assert row["spread_reference_line"] == -3.0
    assert row["spread_play"] == "Clemson"
    assert row["spread_line"] == -2.0
    assert row["spread_provider"] == "DraftKings"
    assert row["spread_diff"] == 4.0
    assert row["spread_shopping_points"] == 1.0
    assert row["total_reference_line"] == 50.0
    assert row["total_play"] == "under"
    assert row["total_line"] == 51.0
    assert row["total_shopping_points"] == 1.0
    assert bool(row["spread_market_supported"]) is True
    assert bool(row["total_market_supported"]) is True


def test_isolated_extreme_is_not_an_execution_candidate():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("DraftKings", -3.0, 50.0),
                        quote("Bovada", -3.5, 50.5),
                        quote("ESPN Bet", 6.0, 64.5),
                    ]
                )
            ]
        )
    )

    bets = scores_to_cfb_bets(with_predictions(markets))
    row = bets.iloc[0]
    assert row["spread_line"] == -3.0
    assert row["spread_provider"] == "DraftKings"
    assert row["total_line"] == 50.5
    assert row["total_provider"] == "Bovada"
    assert row["spread_selection_reason"] == "corroborated_cluster"


def test_disagreeing_two_book_market_uses_reliable_fallback_without_shopping():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("DraftKings", -2.0, 48.0),
                        quote("Bovada", -5.0, 52.0),
                    ]
                )
            ]
        )
    )
    bets = scores_to_cfb_bets(with_predictions(markets))
    row = bets.iloc[0]

    assert row["spread_reference_line"] == -3.5
    assert row["spread_line"] == -5.0
    assert row["spread_provider"] == "Bovada"
    assert row["spread_selection_reason"] == "uncorroborated_fallback"
    assert bool(row["spread_market_supported"]) is False
    assert row["total_line"] == 52.0
    assert bool(row["total_market_supported"]) is False


def test_one_book_and_missing_optional_fields_remain_eligible_but_unsupported():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame([game([quote("DraftKings", -2.5, 48.5)])])
    )
    row = markets.iloc[0]

    assert row["spread_reference_line"] == -2.5
    assert row["total_reference_line"] == 48.5
    assert bool(row["spread_market_supported"]) is False
    assert bool(row["total_market_supported"]) is False
    assert row["consensus_home_win_prob_missing"] == 1
    assert pd.isna(row["spread_open_line"])

    bet = scores_to_cfb_bets(with_predictions(markets)).iloc[0]
    assert bet["spread_line"] == bet["spread_reference_line"] == -2.5
    assert bet["total_line"] == bet["total_reference_line"] == 48.5


def test_provider_order_duplicates_and_synthetic_sources_do_not_change_output():
    lines = [
        quote("Draft Kings", -2.0, 49.0, home_ml=-130, away_ml=110),
        quote("DraftKings", -4.0, 51.0, home_ml=-150, away_ml=120),
        quote("Bovada", -3.0, 50.0, home_ml=-140, away_ml=120),
        quote("consensus", 30.0, 90.0),
        quote("teamrankings", -20.0, 20.0),
    ]
    first = assemble_cfb_betting_markets(pd.DataFrame([game(lines)]))
    second = assemble_cfb_betting_markets(pd.DataFrame([game(list(reversed(lines)))]))

    pd.testing.assert_frame_equal(first, second)
    row = first.iloc[0]
    assert row["spread_reference_line"] == -3.0
    assert row["total_reference_line"] == 50.0
    assert row["spread_quote_count"] == 2
    assert row["total_quote_count"] == 2
    assert row["consensus_home_win_prob_missing"] == 0
    assert 0 < row["consensus_home_win_prob"] < 1


def test_cluster_ties_prefer_consensus_then_provider_reliability():
    consensus_tie = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("A", -10.0, 40.0),
                        quote("B", -9.0, 41.0),
                        quote("C", -2.0, 48.0),
                        quote("D", -1.0, 49.0),
                        quote("E", 3.0, 56.0),
                        quote("F", 4.0, 57.0),
                    ]
                )
            ]
        )
    ).iloc[0]
    assert consensus_tie["spread_execution_candidates"] == (("C", -2.0), ("D", -1.0))
    assert consensus_tie["total_execution_candidates"] == (("C", 48.0), ("D", 49.0))

    reliability_tie = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("Bovada", -5.0, 45.0),
                        quote("Other", -4.0, 46.0),
                        quote("ESPN Bet", 0.0, 50.0),
                        quote("DraftKings", 1.0, 51.0),
                    ]
                )
            ]
        )
    ).iloc[0]
    assert reliability_tie["spread_execution_candidates"] == (
        ("Bovada", -5.0),
        ("Other", -4.0),
    )
    assert reliability_tie["total_execution_candidates"] == (
        ("Bovada", 45.0),
        ("Other", 46.0),
    )


def test_away_and_over_shop_the_lowest_corroborated_quotes():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("Bovada", -3.5, 49.0),
                        quote("DraftKings", -2.0, 50.5),
                        quote("ESPN Bet", -3.0, 50.0),
                    ]
                )
            ]
        )
    )

    row = scores_to_cfb_bets(with_predictions(markets, home=21.0, away=30.0)).iloc[0]
    assert row["spread_play"] == "LSU"
    assert row["spread_line"] == -3.5
    assert row["total_play"] == "over"
    assert row["total_line"] == 49.0


def test_spread_and_total_support_are_independent():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("Bovada", -5.0, 49.0),
                        quote("DraftKings", -2.0, 50.0),
                    ]
                )
            ]
        )
    )
    row = markets.iloc[0]

    assert bool(row["spread_market_supported"]) is False
    assert bool(row["total_market_supported"]) is True


def test_non_fbs_and_incomplete_markets_are_excluded():
    frame = pd.DataFrame(
        [
            game([quote("Bovada", -3.0, 50.0)], game_id=1, home_classification="fcs", away_classification="fcs"),
            game([quote("Bovada", -3.0, None)], game_id=2),
            game([quote("Bovada", None, 50.0)], game_id=3),
            game([quote("Bovada", -3.0, 50.0)], game_id=4, away_classification="fcs"),
        ]
    )

    markets = assemble_cfb_betting_markets(frame)
    assert markets["id"].tolist() == [4]


def test_unsupported_markets_cannot_consume_or_receive_locks():
    bets = pd.DataFrame(
        {
            "game_id": ["unsupported", "supported"],
            "spread_pred": [-10.0, -8.0],
            "spread_line": [-2.0, -2.0],
            "spread_win_prob": [99.0, 70.0],
            "spread_market_supported": [False, True],
            "total_pred": [60.0, 55.0],
            "total_line": [45.0, 45.0],
            "total_win_prob": [99.0, 70.0],
            "total_market_supported": [False, True],
        }
    )
    thresholds = PlayThresholds(max_spreads_plays=1, max_total_plays=1)

    output = determine_plays(bets, thresholds)

    assert output.set_index("game_id").loc["unsupported", "spread_lock"] == 0
    assert output.set_index("game_id").loc["unsupported", "total_lock"] == 0
    assert output.set_index("game_id").loc["supported", "spread_lock"] == 1
    assert output.set_index("game_id").loc["supported", "total_lock"] == 1


def test_missing_required_cfbd_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        assemble_cfb_betting_markets(pd.DataFrame({"id": [1], "lines": [[]]}))


def test_all_ineligible_games_return_a_mergeable_empty_frame():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [quote("Bovada", -3.0, 50.0)],
                    home_classification="fcs",
                    away_classification="fcs",
                )
            ]
        )
    )

    assert markets.empty
    assert "id" in markets.columns


def test_absolute_and_pick_aligned_movement_features_are_available():
    markets = assemble_cfb_betting_markets(
        pd.DataFrame(
            [
                game(
                    [
                        quote("Bovada", -3.0, 50.0, spread_open=-1.0, total_open=47.0),
                        quote("DraftKings", -2.0, 51.0, spread_open=-2.0, total_open=48.0),
                    ]
                )
            ]
        )
    )
    row = scores_to_cfb_bets(with_predictions(markets)).iloc[0]

    assert row["spread_line_move_abs"] == 1.0
    assert row["total_line_move_abs"] == 3.0
    assert row["spread_move_toward_pick"] == 1.0
    assert row["total_move_toward_pick"] == -3.0


def test_optional_confidence_groups_cover_opening_movement_depth_and_no_vig():
    for market in ("spread", "total"):
        groups = OPTIONAL_CONFIDENCE_FEATURE_GROUPS[market]
        assert set(groups) == {"opening", "movement", "market_depth"}
        assert f"{market}_open_line" in groups["opening"]
        assert f"{market}_line_move" in groups["movement"]
        assert f"{market}_line_move_abs" in groups["movement"]
        assert f"{market}_move_toward_pick" in groups["movement"]
        assert f"{market}_quote_count" in groups["market_depth"]
        assert f"{market}_quote_range" in groups["market_depth"]
        assert f"{market}_market_supported" in groups["market_depth"]
        assert "consensus_home_win_prob" in groups["market_depth"]
        assert "consensus_home_win_prob_missing" in groups["market_depth"]
