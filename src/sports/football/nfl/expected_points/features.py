"""Stable NFL expected-points feature calculations."""

from __future__ import annotations

import math


def calculate_nfl_passer_rating(
    attempts: float,
    completions: float,
    passing_yards: float,
    passing_touchdowns: float,
    interceptions: float,
) -> float:
    """Calculate the official NFL passer rating, bounded to 0–158.3."""
    try:
        attempts, completions, passing_yards, passing_touchdowns, interceptions = (
            float(value)
            for value in (
                attempts,
                completions,
                passing_yards,
                passing_touchdowns,
                interceptions,
            )
        )
    except (TypeError, ValueError):
        return float("nan")
    if not all(
        math.isfinite(value)
        for value in (
            attempts,
            completions,
            passing_yards,
            passing_touchdowns,
            interceptions,
        )
    ):
        return float("nan")
    if attempts <= 0:
        return 0.0

    components = (
        ((completions / attempts) - 0.3) * 5,
        ((passing_yards / attempts) - 3) * 0.25,
        (passing_touchdowns / attempts) * 20,
        2.375 - ((interceptions / attempts) * 25),
    )
    bounded = [min(2.375, max(0.0, component)) for component in components]
    return float(sum(bounded) / 6 * 100)


__all__ = ["calculate_nfl_passer_rating"]
