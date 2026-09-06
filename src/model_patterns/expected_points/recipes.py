"""Recipe lookup kept free of notebook and persistence concerns."""

from __future__ import annotations

from .types import ExpectedPointsLeague, ExpectedPointsRecipe


def get_expected_points_recipe(
    league: ExpectedPointsLeague | str,
    *,
    version: str = "working-tree",
) -> ExpectedPointsRecipe:
    parsed = league if isinstance(league, ExpectedPointsLeague) else ExpectedPointsLeague(league)
    if parsed is ExpectedPointsLeague.NFL:
        from src.sports.football.nfl.expected_points.recipe import (
            NFLExpectedPointsRecipe,
        )

        return NFLExpectedPointsRecipe(version=version)
    from src.sports.football.cfb.expected_points.recipe import CFBExpectedPointsRecipe

    return CFBExpectedPointsRecipe(version=version)


__all__ = ["get_expected_points_recipe"]
