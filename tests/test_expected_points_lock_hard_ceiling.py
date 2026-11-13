"""The preferred capacity and absolute limit remain distinct with frozen picks."""
import numpy as np
import pandas as pd
import pytest

from src.model_patterns.expected_points.lock_policy import LockPolicy, add_outcome_probabilities, select_weekly_locks
from src.model_patterns.expected_points.tracking import prepare_tracking_run
from src.model_patterns.expected_points.types import ExpectedPointsTrackingConfig, PlayThresholds


def _policy():
    return LockPolicy(minimum_resolved_win_probability=.525, max_locks_per_week=3,
                      extra_lock_min_probability=.575, hard_max_locks_per_week=10)


def _decisions(count=15, probability=.6):
    frame = pd.DataFrame({'season':2026,'week':1,'game_id':[str(i) for i in range(count)],
                          'market':'spread','kickoff':pd.Timestamp('2026-09-10T17:00Z'),
                          'edge':3.,'market_supported':True,'locks_enabled':True})
    return add_outcome_probabilities(frame,np.full(count,probability),np.zeros(count),policy=_policy())


@pytest.mark.parametrize('preserved_count,remaining',[(9,1),(10,0),(12,0)])
def test_absent_preserved_locks_consume_the_absolute_capacity(preserved_count,remaining):
    frame=_decisions(24)
    preserved=frame.iloc[:preserved_count].assign(lock=1)
    # The selector receives only new decisions; supplied frozen keys still reserve capacity.
    fresh=frame.iloc[preserved_count:].copy()
    result=select_weekly_locks(fresh,policy=_policy(),preserved=preserved)
    assert result.lock.sum()==remaining
    pd.testing.assert_series_equal(result.resolved_win_probability,fresh.resolved_win_probability)


def test_extra_probability_boundary_and_no_quota():
    assert select_weekly_locks(_decisions(5,.575),policy=_policy()).lock.sum()==5
    assert select_weekly_locks(_decisions(5,.574999),policy=_policy()).lock.sum()==3
    assert select_weekly_locks(_decisions(2,.53),policy=_policy()).lock.sum()==2
    assert select_weekly_locks(_decisions(5,.51),policy=_policy()).lock.sum()==0


def _pick(game_id, *, started=False, total_lock=1):
    return {'season':2026,'week':1,'year_week':'2026_1','game_id':game_id,
            'home_team':'A','away_team':'B','home_score_pred':27.,'away_score_pred':20.,
            'spread_pred':-7.,'spread_line':-3.5,'spread_play':'A','spread_win_prob':61.,'spread_lock':1,
            'total_pred':47.,'total_line':44.5,'total_play':'over','total_win_prob':58.,'total_lock':total_lock,
            'date_time':'2026-08-01-00:15' if started else '2026-08-02-13:00',
            'write_time':'2026-07-31 00:00:00'}


@pytest.mark.parametrize('preserved_count,remaining',[(9,1),(10,0)])
def test_tracking_reserves_absent_started_locks_before_admitting_extras(preserved_count,remaining):
    old=[_pick(f'old-{i}',started=True,total_lock=int(2*i+1<preserved_count)) for i in range((preserved_count+1)//2)]
    original=pd.DataFrame(old);unchanged=original.copy(deep=True)
    thresholds=PlayThresholds(max_spreads_plays=None,max_total_plays=None,max_combined_plays=3,
                             extra_lock_min_probability=.575,hard_max_combined_plays=10,
                             min_spread_win_prob=52.5,min_total_win_prob=52.5)
    run=prepare_tracking_run(pd.DataFrame([_pick('new')]),original,'2026_1',
                             ExpectedPointsTrackingConfig(play_thresholds=thresholds),
                             now=pd.Timestamp('2026-08-01T04:00:00Z'))
    assert run.picks.game_id.tolist()==['new']
    assert run.picks[['spread_lock','total_lock']].sum().sum()==remaining
    pd.testing.assert_frame_equal(original,unchanged)
