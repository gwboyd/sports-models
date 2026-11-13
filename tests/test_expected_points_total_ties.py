"""Exact prediction ties retain valid sides without manufacturing model edge."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.model_patterns.expected_points.betting import (
    calculate_wins, directional_bet_edge, scores_to_bets,
)
from src.model_patterns.expected_points.lock_models import SymmetricResidualHead
from src.model_patterns.expected_points.lock_policy import (
    LockPolicy, add_outcome_probabilities, select_weekly_locks,
)
from src.model_patterns.expected_points.lock_replay import build_lock_market_frame
from src.model_patterns.expected_points.tracking import build_pick_records, validate_pick_frame
from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football.cfb.expected_points.betting_lines import (
    assemble_cfb_betting_markets, scores_to_cfb_bets,
)
from src.sports.football.expected_points_schemas import PickResponse


def prediction(league, *, home=24., totals=(44., 44.)):
    common = dict(season=2026, week=1, year_week='2026_1', game_id='tie',
                  home_team='H', away_team='A', date_time='2026-09-12-13:00',
                  home_score_pred=home, away_score_pred=20., spread_win_prob=50.,
                  total_win_prob=50., spread_lock=0, total_lock=0,
                  spread_locks_enabled=True, total_locks_enabled=True)
    if league == 'nfl':
        frame = pd.DataFrame([{**common, 'spread_line': -3.5, 'total_line': 44.,
                               'implied_points_home': 23.75, 'implied_points_away': 20.25}])
        return frame, scores_to_bets
    providers = ('Bovada', 'DraftKings')
    market = assemble_cfb_betting_markets(pd.DataFrame([{
        'id': 'tie', 'home_classification': 'fbs', 'away_classification': 'fbs',
        'lines': [{'provider': provider, 'spread': -3.5, 'overUnder': total}
                  for provider, total in zip(providers, totals)],
    }]))
    return market.assign(**common), scores_to_cfb_bets


@pytest.mark.parametrize('league', ['nfl', 'cfb'])
def test_exact_total_tie_is_persistable_over_with_zero_edge_and_fifty_percent(league):
    frame, adapt = prediction(league)
    original = frame.copy(deep=True)
    picks = adapt(frame)
    assert picks.total_play.item() == 'over'
    assert picks.total_pred.item() == picks.total_line.item() == 44.
    assert directional_bet_edge(picks, 'total').item() == 0.
    pd.testing.assert_frame_equal(frame, original)

    features = build_lock_market_frame(picks, ExpectedPointsLeague(league), 'total', outcomes_known=False)
    assert features.score_residual.isna().all()
    head = SymmetricResidualHead(np.array([0., .25, 1., 3.]), trust=1.)
    q = head.predict(features)
    np.testing.assert_array_equal(q, [.5])

    validated = validate_pick_frame(picks, expected_game_ids=['tie'])
    records = build_pick_records(validated, datetime(2026, 9, 7, tzinfo=timezone.utc))
    assert records[0]['total_play'] == 'over'
    response = PickResponse(**{**records[0], 'write_time': '2026-09-07T00:00:00Z'})
    assert response.total_play == 'over'

    # Selection quotas remain a policy choice; zero edge does not inflate q.
    features = features.assign(locks_enabled=True, policy_eligible=True)
    policy = LockPolicy(minimum_resolved_win_probability=.525, max_locks_per_week=3,
                        extra_lock_min_probability=.57, hard_max_locks_per_week=10)
    decisions = add_outcome_probabilities(features, q, np.zeros(len(q)), policy=policy)
    assert select_weekly_locks(decisions, policy=policy).lock.item() == 0
    assert decisions.expected_units.item() < 0


def test_cfb_consensus_tie_uses_supported_over_execution_and_actual_shopping_edge():
    frame, adapt = prediction('cfb', totals=(43.5, 44.5))
    picks = adapt(frame)
    assert picks.total_reference_line.item() == picks.total_pred.item() == 44.
    assert picks.total_play.item() == 'over'
    assert picks.total_market_supported.item()
    assert picks.total_provider.item() == 'Bovada'
    assert picks.total_line.item() == 43.5
    assert picks.total_diff.item() == picks.total_shopping_points.item() == .5
    features = build_lock_market_frame(picks, ExpectedPointsLeague.CFB, 'total', outcomes_known=False)
    head = SymmetricResidualHead(np.array([.25, 1., 2., 3.]), trust=1.)
    assert head.predict(features).item() == .625  # Only the real half-point edge changes q.

    permuted = frame.copy(deep=True)
    permuted['total_execution_candidates'] = permuted.total_execution_candidates.map(lambda values: tuple(reversed(values)))
    pd.testing.assert_frame_equal(adapt(permuted)[['total_play','total_line','total_diff']],
                                  picks[['total_play','total_line','total_diff']])


@pytest.mark.parametrize('league', ['nfl', 'cfb'])
@pytest.mark.parametrize('home_prediction, side', [(24.-1e-9, 'under'), (24.+1e-9, 'over')])
def test_total_tie_rule_does_not_round_nearby_predictions(league, home_prediction, side):
    frame, adapt = prediction(league, home=home_prediction)
    assert adapt(frame).total_play.item() == side


@pytest.mark.parametrize('league', ['nfl', 'cfb'])
@pytest.mark.parametrize('actual_home, expected_win', [(23., 0), (24., None), (25., 1)])
def test_exact_prediction_tie_is_graded_only_against_final_score(league, actual_home, expected_win):
    frame, adapt = prediction(league)
    baseline = adapt(frame)
    with_outcome = frame.assign(home_score=actual_home, away_score=20.)
    predicted = adapt(with_outcome)
    pd.testing.assert_frame_equal(predicted[['total_play','total_pred','total_line','total_diff']],
                                  baseline[['total_play','total_pred','total_line','total_diff']])
    assert predicted.total_play.item() == 'over'
    result = calculate_wins(predicted).iloc[0]
    if expected_win is None:
        assert pd.isna(result.total_win)
        assert pd.isna(result.correct_total_play)
    else:
        assert result.total_win == expected_win


@pytest.mark.parametrize('league', ['nfl', 'cfb'])
def test_nonfinite_prediction_is_not_repaired_by_exact_tie_rule(league):
    frame, adapt = prediction(league, home=np.nan)
    picks = adapt(frame)
    assert pd.isna(picks.total_play.item())
    with pytest.raises(ValueError, match='null required values'):
        validate_pick_frame(picks)
