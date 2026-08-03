from .types import (
    ExpectedPointsConfig,
    ExpectedPointsLeague,
    ExpectedPointsRunResult,
    ExpectedPointsTrackingConfig,
    PlayThresholds,
)
from .reporting import get_feature_importance_df, get_result_stats, print_plays, summarize_eval_results


def __getattr__(name: str):
    """Keep training dependencies out of API-only process startup."""
    if name == "run_expected_points":
        from .trainer import run_expected_points

        return run_expected_points
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExpectedPointsConfig",
    "ExpectedPointsLeague",
    "ExpectedPointsRunResult",
    "ExpectedPointsTrackingConfig",
    "PlayThresholds",
    "get_feature_importance_df",
    "get_result_stats",
    "print_plays",
    "summarize_eval_results",
    "run_expected_points",
]
