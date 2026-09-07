"""CFB history policy excludes COVID before fitting and rating updates."""
import pandas as pd
import pytest

from src.sports.football.cfb.expected_points.history import CFBHistoryConfig


def test_default_keeps_2017_warmup_and_2018_training_but_excludes_covid():
    config = CFBHistoryConfig()
    assert config.loading_years(2022) == (2017, 2018, 2019, 2021, 2022)
    games = pd.DataFrame({'season': [2016, 2017, 2018, 2019, 2020, 2021, 2022]})
    assert config.filter_games(games).season.tolist() == [2018, 2019, 2021, 2022]
    assert config.filter_games(games, training_only=False).season.tolist() == [2017, 2018, 2019, 2021, 2022]
    assert len(games) == 7


def test_history_overrides_are_independent_and_validated():
    config = CFBHistoryConfig(training_start_season=2022, feature_start_season=2018,
                              excluded_seasons=(2019, 2020))
    assert config.loading_years(2023) == (2018, 2021, 2022, 2023)
    with pytest.raises(ValueError, match='after training'):
        CFBHistoryConfig(feature_start_season=2021)
    with pytest.raises(ValueError, match='integer'):
        CFBHistoryConfig(training_start_season=True)
    with pytest.raises(ValueError, match='precedes'):
        CFBHistoryConfig().loading_years(2017)
