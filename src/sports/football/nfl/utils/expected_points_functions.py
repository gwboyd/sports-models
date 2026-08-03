"""Compatibility exports for shared expected-points helpers."""

from src.model_patterns.expected_points.betting import calculate_wins
from src.model_patterns.expected_points.reporting import get_result_stats as _get_result_stats


def get_result_stats(df, Verbose=False):
    return _get_result_stats(df, verbose=Verbose)


__all__ = ["calculate_wins", "get_result_stats"]
