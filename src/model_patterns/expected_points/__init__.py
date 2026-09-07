from .inspection import (
    health_metrics_frame,
    inspect_game,
    inspect_team_history,
    load_backtest_comparison,
)
from .recipes import get_expected_points_recipe
from .reporting import (
    get_feature_importance_df,
    get_result_stats,
    print_plays,
    summarize_eval_results,
)
from .types import (
    ExpectedPointsConfig,
    ExpectedPointsLeague,
    ExpectedPointsRecipe,
    ExpectedPointsRunResult,
    ExpectedPointsTrackingConfig,
    LockHeadConfig,
    PlayThresholds,
)
from .versioning import (
    ModelKey,
    ModelVersion,
    ReleaseDraft,
    next_major,
    next_minor,
    parse_release_file,
    parse_release_markdown,
    parse_version,
)


def __getattr__(name: str):
    """Keep training dependencies out of API-only process startup."""
    if name == "run_expected_points":
        from .trainer import run_expected_points

        return run_expected_points
    if name in {
        "BacktestComparison",
        "BacktestRun",
        "BacktestSpec",
        "bootstrap_metric_intervals",
        "compare_backtest_runs",
        "load_backtest_run",
        "run_walk_forward",
        "save_backtest_frame",
        "write_comparison_artifacts",
        "write_run_artifacts",
    }:
        from . import backtesting

        return getattr(backtesting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BacktestComparison",
    "BacktestRun",
    "BacktestSpec",
    "ExpectedPointsConfig",
    "ExpectedPointsLeague",
    "ExpectedPointsRecipe",
    "ExpectedPointsRunResult",
    "ExpectedPointsTrackingConfig",
    "LockHeadConfig",
    "ModelKey",
    "ModelVersion",
    "PlayThresholds",
    "ReleaseDraft",
    "bootstrap_metric_intervals",
    "compare_backtest_runs",
    "get_expected_points_recipe",
    "get_feature_importance_df",
    "get_result_stats",
    "health_metrics_frame",
    "inspect_game",
    "inspect_team_history",
    "load_backtest_comparison",
    "load_backtest_run",
    "next_major",
    "next_minor",
    "parse_release_file",
    "parse_release_markdown",
    "parse_version",
    "print_plays",
    "run_expected_points",
    "run_walk_forward",
    "save_backtest_frame",
    "summarize_eval_results",
    "write_comparison_artifacts",
    "write_run_artifacts",
]
