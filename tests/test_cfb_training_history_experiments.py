"""Training-window comparisons cannot change retained features or evaluated games."""
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.experiment_cfb_training_history import (
    compare_training_windows, validate_history_extension,
)
from src.model_patterns.expected_points.backtesting import BacktestSpec


def frames():
    frame = pd.DataFrame({
        'game_id': ['old', 'warmup', 'evaluation'], 'season': [2018, 2022, 2023],
        'week': [1, 1, 1], 'date_time': ['2018-09-01-12:00', '2022-09-01-12:00', '2023-09-01-12:00'],
        'home_team': ['A'] * 3, 'away_team': ['B'] * 3,
        'home_score': [20, 21, 22], 'away_score': [10] * 3,
        'spread_line': [-3.] * 3, 'total_line': [40.] * 3,
        'feature': [0.5, 0.6, 0.7],
    })
    return frame.iloc[1:].copy(), frame


def test_allows_only_earlier_training_rows_with_fixed_inputs():
    reference, extended = frames()
    design = validate_history_extension(reference, extended, BacktestSpec(seasons=(2023,)))
    assert design['added_rows_by_season'] == {2018: 1}
    assert design['evaluation_rows'] == 1
    assert design['reference_source_fingerprint'] != design['candidate_source_fingerprint']


@pytest.mark.parametrize('column,value', [('feature', 9.), ('home_score', 99), ('spread_line', -9.)])
def test_rejects_any_changed_retained_input(column, value):
    reference, extended = frames()
    extended.loc[1, column] = value
    with pytest.raises(ValueError, match='changed existing'):
        validate_history_extension(reference, extended, BacktestSpec(seasons=(2023,)))


def test_rejects_covid_rows_and_late_additions():
    reference, extended = frames()
    with pytest.raises(ValueError, match='Excluded seasons'):
        validate_history_extension(reference, extended.assign(season=[2020, 2022, 2023]),
                                   BacktestSpec(seasons=(2023,)))
    extended.loc[0, 'date_time'] = '2023-01-01-12:00'
    with pytest.raises(ValueError, match='must precede'):
        validate_history_extension(reference, extended, BacktestSpec(seasons=(2023,)))


def test_comparison_rejects_unrelated_run_artifacts():
    reference, extended = frames()
    design = validate_history_extension(reference, extended, BacktestSpec(seasons=(2023,)))
    unrelated = SimpleNamespace(source_fingerprint='unrelated')
    with pytest.raises(ValueError, match='fingerprints'):
        compare_training_windows(unrelated, unrelated, design)
