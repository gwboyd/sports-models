"""Quarterback features must not learn debut priors from later outcomes."""
import pandas as pd
import pytest

from src.sports.football.nfl.expected_points.features import build_pregame_quarterback_metrics


def history():
    return pd.DataFrame({
        'player_id': ['A', 'A', 'A', 'B', None],
        'player_name': ['Alpha', 'Alpha', 'Alpha', 'Beta', None],
        'season': [2025] * 5, 'week': [1, 2, 3, 3, 3],
        'gameday': ['2025-09-01', '2025-09-08', '2025-09-15', '2025-09-15', '2025-09-15'],
        'gametime': ['13:00'] * 5, 'team': ['X', 'X', 'X', 'Y', 'Z'],
        'qbr': [80., 100., 150., 140., float('nan')],
        'passer_rating': [80., 100., 150., 140., float('nan')],
    })


def test_debut_and_unknown_starter_remain_missing_and_returning_starter_uses_past():
    result = build_pregame_quarterback_metrics(history())
    assert pd.isna(result.loc[(result.player_id == 'A') & (result.week == 1), 'ewma_qbr'].iloc[0])
    assert result.loc[(result.player_id == 'A') & (result.week == 2), 'ewma_qbr'].iloc[0] == 80.
    assert pd.isna(result.loc[result.player_id == 'B', 'ewma_qbr'].iloc[0])
    assert result.loc[result.player_id.isna(), 'ewma_qbr'].isna().all()


def test_future_and_target_outcomes_cannot_change_an_earlier_quarterback_feature():
    original = history()
    changed = original.copy()
    changed.loc[changed.week >= 2, ['qbr', 'passer_rating']] = 9999.
    before = build_pregame_quarterback_metrics(original)
    after = build_pregame_quarterback_metrics(changed.sample(frac=1, random_state=31))
    cols = ['player_id', 'week', 'ewma_qbr', 'ewma_passer_rating']
    pd.testing.assert_frame_equal(before.loc[before.week <= 2, cols].reset_index(drop=True),
                                  after.loc[after.week <= 2, cols].reset_index(drop=True))
    truncated = build_pregame_quarterback_metrics(original.loc[original.week <= 2])
    pd.testing.assert_frame_equal(before.loc[before.week <= 2, cols].reset_index(drop=True),
                                  truncated[cols].reset_index(drop=True))


def test_quarterback_chronology_rejects_ambiguous_starts():
    with pytest.raises(ValueError, match='same kickoff'):
        build_pregame_quarterback_metrics(pd.concat([history(), history().iloc[:1]]))
