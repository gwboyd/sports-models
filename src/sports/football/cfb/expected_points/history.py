"""Explicit CFB training and feature-history boundaries."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CFBHistoryConfig:
    training_start_season: int = 2018
    excluded_seasons: tuple[int, ...] = (2020,)
    feature_start_season: int | None = None

    def __post_init__(self) -> None:
        years = (self.training_start_season, *self.excluded_seasons)
        if self.feature_start_season is not None:
            years += (self.feature_start_season,)
        if any(isinstance(year, bool) or not isinstance(year, int) or year < 1869 for year in years):
            raise ValueError('CFB history seasons must be integer years at or after 1869')
        if self.feature_start > self.training_start_season:
            raise ValueError('CFB feature history cannot start after training history')

    @property
    def feature_start(self) -> int:
        return self.training_start_season - 1 if self.feature_start_season is None else self.feature_start_season

    def loading_years(self, through_season: int) -> tuple[int, ...]:
        if through_season < self.training_start_season:
            raise ValueError('CFB history end precedes its training start')
        return tuple(year for year in range(self.feature_start, through_season + 1)
                     if year not in self.excluded_seasons)

    def filter_games(self, games: pd.DataFrame, *, training_only: bool = True) -> pd.DataFrame:
        """Exclude seasons before any Elo update or opponent-feature calculation."""
        seasons = pd.to_numeric(games['season'], errors='raise')
        start = self.training_start_season if training_only else self.feature_start
        return games.loc[seasons.ge(start) & ~seasons.isin(self.excluded_seasons)].copy()


DEFAULT_CFB_HISTORY = CFBHistoryConfig()
