"""Explicit selection minimums preserve probabilities, safeguards and capacity."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.model_patterns.expected_points.betting import determine_plays
from src.model_patterns.expected_points.lock_policy import LockPolicy, add_outcome_probabilities, select_weekly_locks
from src.model_patterns.expected_points.types import PlayThresholds


def _policy(**changes):
    return replace(LockPolicy(minimum_resolved_win_probability=.525, min_locks_per_week=3,
                              max_locks_per_week=3, extra_lock_min_probability=.57,
                              hard_max_locks_per_week=10), **changes)


def _decisions(probabilities):
    d = pd.DataFrame({'season': 2026, 'week': 1, 'game_id': [f'g{i}' for i in range(len(probabilities))],
                      'market': 'spread', 'kickoff': pd.Timestamp('2026-09-10T17:00Z'),
                      'edge': 1., 'market_supported': True, 'locks_enabled': True})
    return add_outcome_probabilities(d, np.array(probabilities), np.zeros(len(d)), policy=_policy())


def test_minimum_admits_top_probabilities_without_changing_negative_ev_forecasts():
    d = _decisions([.51, .502, .509, .505, .508])
    before = d.copy(deep=True)
    # Expected-profit ordering differs; minimum selection is probability-first.
    d.loc[1, 'expected_units'] = -.001
    r = select_weekly_locks(d, policy=_policy(extra_lock_min_probability=None, hard_max_locks_per_week=3))
    assert r.loc[r.lock.eq(1), 'game_id'].tolist() == ['g0', 'g2', 'g4']
    assert r.loc[r.lock.eq(1), 'lock_reason'].eq('selected_to_minimum').all()
    pd.testing.assert_series_equal(r.resolved_win_probability, before.resolved_win_probability)
    pd.testing.assert_series_equal(r.expected_units, d.expected_units)


@pytest.mark.parametrize('q,count', [(.57, 10), (.569999, 3), (.51, 3)])
def test_minimum_and_exact_extra_threshold_respect_hard_ceiling(q, count):
    assert select_weekly_locks(_decisions([q] * 15), policy=_policy()).lock.sum() == count


def test_minimum_never_overrides_support_readiness_invalid_forecasts_or_market_disable():
    d = _decisions([.6] * 7)
    d['policy_eligible'] = True
    d.loc[0, 'market_supported'] = False
    d.loc[1, 'locks_enabled'] = False
    d.loc[2, 'resolved_win_probability'] = np.nan
    d.loc[3, 'edge'] = -1
    d.loc[4, 'policy_eligible'] = False
    d.loc[5, 'market'] = 'total'
    r = select_weekly_locks(d, policy=_policy(max_totals_per_week=0))
    assert r.loc[r.lock.eq(1), 'game_id'].tolist() == ['g6']


@pytest.mark.parametrize('reserved,remaining', [(1, 2), (3, 0), (5, 0)])
def test_absent_preserved_locks_consume_minimum_and_nfl_cap(reserved, remaining):
    d = _decisions([.51] * 12)
    frozen = d.iloc[:reserved].assign(lock=1)
    r = select_weekly_locks(d.iloc[reserved:], policy=_policy(extra_lock_min_probability=None,
                             hard_max_locks_per_week=3), preserved=frozen)
    assert r.lock.sum() == remaining


def test_cfb_preserved_minimum_still_allows_only_qualified_extras():
    d = _decisions([.51, .51, .51, .57, .5699, .58])
    r = select_weekly_locks(d.iloc[3:], policy=_policy(), preserved=d.iloc[:3].assign(lock=1))
    assert r.loc[r.lock.eq(1), 'game_id'].tolist() == ['g3', 'g5']


@pytest.mark.parametrize('changes', [{'min_locks_per_week': -1}, {'min_locks_per_week': 3.5},
                                    {'min_locks_per_week': 4}, {'hard_max_locks_per_week': 2}])
def test_invalid_minimum_fails(changes):
    with pytest.raises(ValueError, match='minimum'):
        _policy(**changes)


def test_production_minimum_bypasses_only_probability_floor_and_ev_and_resets_weekly():
    d = pd.DataFrame({'season': 2026, 'week': [1, 1, 2], 'game_id': ['a', 'b', 'c'],
                      'home_team': 'A', 'away_team': 'B', 'spread_line': -3., 'spread_pred': -4.,
                      'total_line': 44., 'total_pred': 45., 'spread_play': 'A', 'total_play': 'over',
                      'spread_win_prob': [51., 50.9, 51.], 'total_win_prob': [50.8, 50.7, 50.8]})
    t = PlayThresholds(min_combined_plays=3, max_combined_plays=3, hard_max_combined_plays=3,
                       max_spreads_plays=None, max_total_plays=None)
    result = determine_plays(d, t)
    assert result.groupby('week')[['spread_lock', 'total_lock']].sum().sum(axis=1).tolist() == [3, 2]
    assert result.spread_win_prob.tolist() == d.spread_win_prob.tolist()
    d.loc[0, 'total_line'] = np.nan
    d.loc[1, 'spread_locks_enabled'] = False
    d['spread_locks_enabled'] = d.spread_locks_enabled.fillna(True)
    assert determine_plays(d, t).loc[:1, ['spread_lock', 'total_lock']].sum().sum() == 2
