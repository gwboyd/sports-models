import pandas as pd
import pytest

from src.sports.football.transforms import (
    OpponentAdjustmentConfig,
    OpponentMetricSpec,
    build_opponent_adjusted_team_metrics,
)
from src.sports.football.nfl.expected_points.features import (
    build_nfl_opponent_adjusted_metrics,
)


SPEC = OpponentMetricSpec("efficiency", "off_efficiency", "def_efficiency", "static")


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4", "5"],
            "season": [2025] * 5,
            "week": [1, 1, 2, 2, 3],
            "home_team": ["A", "C", "B", "C", "A"],
            "away_team": ["B", "B", "A", "A", "C"],
            "start_date": [
                "2025-09-01T12:00:00Z",
                "2025-09-01T13:00:00Z",
                "2025-09-08T12:00:00Z",
                "2025-09-08T13:00:00Z",
                "2025-09-15T12:00:00Z",
            ],
        }
    )


def _observations() -> pd.DataFrame:
    # B allows two strong offensive performances; A's performance against B
    # should therefore receive less credit than its raw value of ten.
    return pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4"],
            "team": ["A", "C", "B", "C"],
            "metric": ["efficiency"] * 4,
            "value": [10.0, 10.0, 0.0, 0.0],
        }
    )


def test_opponent_adjustment_reduces_credit_against_permissive_defense():
    frame = build_opponent_adjusted_team_metrics(_schedule(), _observations(), [SPEC])
    target = frame.loc[(frame["game_id"] == "5") & (frame["team"] == "A")].iloc[0]

    assert target["off_efficiency_ewma"] < 10.0
    assert target["off_efficiency_ewma"] == pytest.approx(8.658536585, abs=1e-8)


def test_future_observation_cannot_change_an_earlier_target():
    schedule = _schedule()
    baseline = build_opponent_adjusted_team_metrics(schedule, _observations(), [SPEC])
    future_schedule = pd.concat(
        [
            schedule,
            pd.DataFrame(
                {
                    "game_id": ["6"], "season": [2025], "week": [4],
                    "home_team": ["B"], "away_team": ["C"],
                    "start_date": ["2025-09-22T12:00:00Z"],
                }
            ),
        ],
        ignore_index=True,
    )
    future_observations = pd.concat(
        [
            _observations(),
            pd.DataFrame({"game_id": ["6"], "team": ["B"], "metric": ["efficiency"], "value": [999.0]}),
        ],
        ignore_index=True,
    )
    with_future = build_opponent_adjusted_team_metrics(future_schedule, future_observations, [SPEC])

    old = baseline.loc[(baseline["game_id"] == "5") & (baseline["team"] == "A"), "off_efficiency_ewma"].iloc[0]
    new = with_future.loc[(with_future["game_id"] == "5") & (with_future["team"] == "A"), "off_efficiency_ewma"].iloc[0]
    assert new == pytest.approx(old)


def test_fcs_pooled_policy_is_a_valid_configurable_alternative():
    schedule = _schedule().assign(
        home_classification=["fbs", "fbs", "fcs", "fbs", "fbs"],
        away_classification=["fcs", "fcs", "fbs", "fcs", "fbs"],
    )
    frame = build_opponent_adjusted_team_metrics(
        schedule,
        _observations(),
        [SPEC],
        config=OpponentAdjustmentConfig(fcs_policy="pooled"),
    )
    assert frame.loc[(frame["game_id"] == "5") & (frame["team"] == "A"), "off_efficiency_ewma"].notna().all()


def test_config_mapping_supports_notebook_parameterization():
    frame = build_opponent_adjusted_team_metrics(
        _schedule(),
        _observations(),
        [SPEC],
        config={"ridge_alpha": 10.0, "season_carryover": 0.25},
    )

    value = frame.loc[
        (frame["game_id"] == "5") & (frame["team"] == "A"), "off_efficiency_ewma"
    ].iloc[0]
    assert pd.notna(value)


def test_nfl_adapter_keeps_the_legacy_model_feature_names():
    schedule = _schedule().rename(columns={"start_date": "kickoff"})
    schedule["gameday"] = pd.to_datetime(schedule["kickoff"]).dt.strftime("%Y-%m-%d")
    schedule["gametime"] = pd.to_datetime(schedule["kickoff"]).dt.strftime("%H:%M")
    pbp = pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4"],
            "season": [2025] * 4,
            "week": [1, 1, 2, 2],
            "posteam": ["A", "C", "B", "C"],
            "defteam": ["B", "B", "A", "A"],
            "rush_attempt": [1, 1, 1, 1],
            "pass_attempt": [0, 0, 0, 0],
            "epa": [0.2, 0.1, 0.0, -0.1],
            "down": [1, 1, 1, 1],
            "yards_gained": [5, 5, 0, 0],
            "ydstogo": [10, 10, 10, 10],
        }
    )

    epa, success = build_nfl_opponent_adjusted_metrics(pbp, schedule)

    assert "ewma_dynamic_window_rushing_offense" in epa
    assert "ewma_success_rate_rushing_offense" in success


@pytest.mark.parametrize("history_week,target_week", [(10, 11), (14, 17), (18, 1)])
def test_dynamic_smoothing_matches_existing_shifted_smoother(history_week, target_week):
    from types import SimpleNamespace
    from src.sports.football.transforms.common import dynamic_period_ewma
    from src.sports.football.transforms.opponent_adjustment import _smooth

    values = [1.0, 8.0, -2.0]
    history = pd.DataFrame({
        "game_id": ["a", "b", "c"], "start_date": pd.date_range("2025-09-01", periods=3),
        "week": [1, 2, history_week], "adjusted": values,
    })
    # The established transform shifts observations onto the next game before
    # choosing that game's span. This includes byes and the new-season reset.
    timeline = pd.DataFrame({"week": [1, 2, history_week, target_week], "shifted": [float("nan"), *values]})
    expected = dynamic_period_ewma(timeline, "shifted").iloc[-1]
    actual = _smooth(history, SimpleNamespace(week=target_week),
                     OpponentMetricSpec("x", "off", "def", "dynamic"), "off")
    assert actual["off_ewma_dynamic_window"] == pytest.approx(expected)


def test_strength_endpoints_and_interpolation_are_exact():
    import numpy as np
    frames = [build_opponent_adjusted_team_metrics(
        _schedule(), _observations(), [SPEC],
        config=OpponentAdjustmentConfig(adjustment_strength=strength),
    ) for strength in (0.0, 0.5, 1.0)]
    column = 'off_efficiency_ewma'
    assert np.allclose(frames[1][column], (frames[0][column] + frames[2][column]) / 2, equal_nan=True)
    assert frames[0].loc[(frames[0].game_id == '5') & (frames[0].team == 'A'), column].iloc[0] == 10.0


def test_zero_carryover_before_season_has_no_weighted_history():
    from src.sports.football.transforms.opponent_adjustment import _attach_opponents, _fit_snapshot
    history = _attach_opponents(_observations(), _schedule())
    assert _fit_snapshot(history, 2026, OpponentAdjustmentConfig(season_carryover=0)) == ({}, {})


def test_excluded_seasons_cannot_influence_ratings_or_smoothing():
    result = build_opponent_adjusted_team_metrics(
        _schedule(), _observations(), [SPEC],
        config=OpponentAdjustmentConfig(excluded_seasons=(2025,)),
    )
    assert result['off_efficiency_ewma'].isna().all()


def test_cfb_registry_builds_all_requested_metrics_and_ignores_mirrored_defense_values():
    from src.sports.football.cfb.expected_points.features import build_opponent_adjusted_advanced_stats
    schedule = _schedule()
    source = _observations().drop(columns=['metric']).merge(
        schedule[['game_id', 'season', 'week', 'start_date']], on='game_id', validate='one_to_one',
    ).rename(columns={'value': 'offense_explosiveness'})
    source['offense_ppa'] = source.offense_explosiveness / 10
    source['defense_ppa'] = 999999.0
    frame = build_opponent_adjusted_advanced_stats(source, schedule, metrics=('explosiveness', 'ppa'))
    assert 'offense_ppa_ewma_dynamic_window' in frame
    assert 'defense_ppa_ewma_dynamic_window' in frame
    target = frame.loc[(frame.game_id == '5') & (frame.team == 'A')].iloc[0]
    assert target.offense_ppa_ewma_dynamic_window == pytest.approx(target.offense_explosiveness_ewma_dynamic_window / 10)
    assert abs(target.defense_ppa_ewma_dynamic_window) < 2


@pytest.mark.parametrize('config', [{'adjustment_strength': -0.1}, {'adjustment_strength': 1.1}, {'fcs_policy': 'typo'}])
def test_invalid_adjustment_levers_fail(config):
    with pytest.raises(ValueError):
        OpponentAdjustmentConfig(**config)


def test_league_mapping_overrides_preserve_unmentioned_defaults():
    from src.sports.football.transforms import resolve_opponent_adjustment_config
    from src.sports.football.cfb.expected_points.features import DEFAULT_CFB_OPPONENT_ADJUSTMENT
    config = resolve_opponent_adjustment_config({'ridge_alpha': 12}, defaults=DEFAULT_CFB_OPPONENT_ADJUSTMENT)
    assert config.ridge_alpha == 12
    assert config.adjustment_strength == .5
    assert config.fcs_policy == 'pooled'
    assert config.excluded_seasons == (2020,)


def test_cfb_warmup_schedule_supplies_pregame_history_without_changing_score_schedule():
    from src.sports.football.cfb.expected_points.features import (
        build_cfb_efficiency_schedule, build_opponent_adjusted_advanced_stats,
    )
    target = pd.DataFrame(dict(game_id=['new'],season=[2022],week=[1],home_team=['A'],away_team=['B'],
        start_date=['2022-09-01T12:00Z'],home_classification=['fbs'],away_classification=['fcs']))
    prior = pd.DataFrame(dict(id=['old'],season=[2021],week=[12],home_team=['A'],away_team=['B'],
        start_date=['2021-11-01T12:00Z'],home_classification=['fbs'],away_classification=['fcs']))
    schedule = build_cfb_efficiency_schedule(target, prior, history_start_season=2021)
    assert target.game_id.tolist() == ['new']
    source = pd.DataFrame(dict(game_id=['old','old','new'],season=[2021,2021,2022],week=[12,12,1],
        team=['A','B','A'],start_date=['2021-11-01T12:00Z']*2+['2022-09-01T12:00Z'],
        offense_explosiveness=[1.,.2,999.]))
    result = build_opponent_adjusted_advanced_stats(source, schedule,
        metrics=('explosiveness',), config={'adjustment_strength':0})
    row = result.loc[(result.game_id=='new') & (result.team=='A')].iloc[0]
    assert row.offense_explosiveness_ewma_dynamic_window == 1.
    assert row.defense_explosiveness_ewma_dynamic_window == .2


def test_smoothing_span_override_changes_recency_without_using_target_outcome():
    source = pd.DataFrame({'game_id':['1','3','5'],'team':['A']*3,
                          'metric':['efficiency']*3,'value':[1.,9.,999.]})
    result = build_opponent_adjusted_team_metrics(_schedule(),source,[SPEC],
        config=OpponentAdjustmentConfig(adjustment_strength=0,smoothing_span=1))
    value = result.loc[(result.game_id=='5') & (result.team=='A'),'off_efficiency_ewma'].iloc[0]
    assert value == 9.
    dynamic = build_opponent_adjusted_team_metrics(_schedule(),source,
        [OpponentMetricSpec('efficiency','off_efficiency','def_efficiency','dynamic')],
        config=OpponentAdjustmentConfig(adjustment_strength=0,smoothing_span=1))
    actual = dynamic.loc[(dynamic.game_id=='5') & (dynamic.team=='A'),'off_efficiency_ewma_dynamic_window'].iloc[0]
    assert actual == pytest.approx(pd.Series([1.,9.]).ewm(span=3).mean().iloc[-1])


@pytest.mark.parametrize('span',[0,-1,1.5,True])
def test_invalid_smoothing_span_is_rejected(span):
    with pytest.raises(ValueError,match='smoothing_span'):
        OpponentAdjustmentConfig(smoothing_span=span)
