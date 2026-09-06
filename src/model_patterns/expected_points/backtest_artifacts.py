"""Persistent, human-readable artifacts for expected-points backtests."""

from __future__ import annotations

import json
import math
import numbers
import platform
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .types import ExpectedPointsLeague

if TYPE_CHECKING:
    from .backtesting import BacktestComparison, BacktestRun


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', value.__class__.__name__)}"
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize ambiguous object columns without changing model columns."""
    output = frame.copy()
    for column in output.select_dtypes(include=["object"]).columns:
        values = output[column].dropna()
        types = {type(value) for value in values}
        if not types or types <= {str}:
            continue
        if any(value_type in {dict, list, tuple, set} for value_type in types):
            output[column] = output[column].map(
                lambda value: None
                if value is None
                else json.dumps(value, sort_keys=True, default=str)
            )
            continue
        if all(issubclass(value_type, (bool, np.bool_)) for value_type in types):
            output[column] = output[column].astype("boolean")
            continue
        if all(
            issubclass(value_type, numbers.Number)
            and not issubclass(value_type, (bool, np.bool_))
            for value_type in types
        ):
            output[column] = pd.to_numeric(output[column], errors="coerce")
            continue
        output[column] = output[column].map(
            lambda value: (
                json.dumps(value, sort_keys=True, default=str)
                if isinstance(value, (dict, list, tuple, set))
                else (None if pd.isna(value) else str(value))
            )
        )
    return output


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        parquet_safe_frame(frame).to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_object_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=["object"]).columns
        if frame[column].dropna().map(
            lambda value: isinstance(value, (dict, list, tuple, set))
        ).any()
    ]


def restore_json_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output:
            output[column] = output[column].map(
                lambda value: None if pd.isna(value) else json.loads(value)
            )
    return output


def _git_identity() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _display_metric(value: Any, *, digits: int = 3) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{parsed:.{digits}f}" if np.isfinite(parsed) else "—"


def _display_interval(interval: tuple[float, float] | None) -> str:
    if interval is None or not all(np.isfinite(value) for value in interval):
        return "—"
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _summary_values(
    summary: pd.DataFrame,
    cut_type: str,
    cut_value: str,
) -> dict[str, Any]:
    selected = summary.loc[
        (summary["cut_type"] == cut_type)
        & (summary["cut_value"].astype(str) == str(cut_value))
    ]
    return dict(zip(selected["metric"], selected["value"]))


def _record(values: dict[str, Any], market: str, *, locks: bool) -> str:
    prefix = f"{market}_lock" if locks else market
    wins = int(values.get(f"{prefix}_wins", 0) or 0)
    losses = int(values.get(f"{prefix}_losses", 0) or 0)
    pushes = int(values.get(f"{prefix}_pushes", 0) or 0)
    percentage = _display_metric(values.get(f"{prefix}_win_pct"), digits=1)
    suffix = f"{percentage}%" if percentage != "—" else "—"
    return f"{wins}-{losses}-{pushes} ({suffix})"


def render_run_report(run: BacktestRun, *, bootstrap_samples: int = 2000) -> str:
    from .backtesting import bootstrap_metric_intervals

    overall = _summary_values(run.summary, "overall", "all")
    intervals = bootstrap_metric_intervals(run.predictions, samples=bootstrap_samples)
    headline = (
        "score_mae", "market_score_mae", "score_mae_advantage_vs_market",
        "margin_mae", "market_margin_mae", "margin_mae_advantage_vs_market",
        "total_points_mae", "market_total_points_mae",
        "total_points_mae_advantage_vs_market", "spread_win_pct",
        "spread_lock_win_pct", "spread_brier", "spread_log_loss", "spread_auc",
        "spread_lock_mean_win_prob", "spread_lock_calibration_gap", "total_win_pct",
        "total_lock_win_pct", "total_brier", "total_log_loss", "total_auc",
        "total_lock_mean_win_prob", "total_lock_calibration_gap",
    )
    lines = [
        "# Expected-Points Historical Run", "",
        f"League: {run.league.value.upper()}", "",
        f"Recipe: `{run.recipe_name}` `{run.recipe_version}`", "",
        f"Profile: `{run.profile}`; seasons: {', '.join(map(str, run.seasons))}", "",
        (
            "> Weekly pre-first-kickoff approximation using currently available historical "
            "feed values; not an exact intraday line or roster snapshot replay."
        ), "",
        "## Records", "",
        "| Cut | Spread | Spread locks | Total | Total locks |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Overall | {_record(overall, 'spread', locks=False)} | "
            f"{_record(overall, 'spread', locks=True)} | "
            f"{_record(overall, 'total', locks=False)} | "
            f"{_record(overall, 'total', locks=True)} |"
        ),
    ]
    for season in run.seasons:
        values = _summary_values(run.summary, "season", str(season))
        lines.append(
            f"| {season} | {_record(values, 'spread', locks=False)} | "
            f"{_record(values, 'spread', locks=True)} | "
            f"{_record(values, 'total', locks=False)} | "
            f"{_record(values, 'total', locks=True)} |"
        )
    lines.extend([
        "", "## Overall diagnostics", "",
        (
            "Positive `*_advantage_vs_market` values mean the model beat the market "
            "baseline on MAE. AUC near 0.500 indicates no useful confidence ranking."
        ), "",
        "| Metric | Value | 95% season-week interval |",
        "|---|---:|---:|",
    ])
    for metric in headline:
        if metric in overall:
            lines.append(
                f"| {metric} | {_display_metric(overall[metric])} | "
                f"{_display_interval(intervals.get(metric))} |"
            )
    return "\n".join(lines) + "\n"


def write_comparison_artifacts(
    comparison: BacktestComparison,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    from .backtesting import DEFAULT_CACHE_ROOT, dependency_fingerprint

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            DEFAULT_CACHE_ROOT / "comparisons" / comparison.baseline.league.value / stamp
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(root / "baseline_predictions.parquet", comparison.baseline.predictions)
    write_parquet_atomic(root / "candidate_predictions.parquet", comparison.candidate.predictions)
    write_json_atomic(root / "summary.json", comparison.summary.to_dict(orient="records"))
    write_json_atomic(root / "deltas.json", comparison.deltas.to_dict(orient="records"))
    write_parquet_atomic(root / "lock_changes.parquet", comparison.lock_changes)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "league": comparison.baseline.league.value,
        "git_identity": _git_identity(),
        "python": platform.python_version(),
        "dependency_fingerprint": dependency_fingerprint(),
        "source_fingerprint": comparison.baseline.source_fingerprint,
        "baseline_json_columns": json_object_columns(comparison.baseline.predictions),
        "candidate_json_columns": json_object_columns(comparison.candidate.predictions),
        "lock_change_json_columns": json_object_columns(comparison.lock_changes),
        "baseline": {
            "name": comparison.baseline.recipe_name,
            "version": comparison.baseline.recipe_version,
            "fingerprint": comparison.baseline.recipe_fingerprint,
            "seasons": comparison.baseline.seasons,
        },
        "candidate": {
            "name": comparison.candidate.recipe_name,
            "version": comparison.candidate.recipe_version,
            "fingerprint": comparison.candidate.recipe_fingerprint,
            "seasons": comparison.candidate.seasons,
        },
        "historical_fidelity": (
            "Weekly pre-first-kickoff replay using currently available historical feeds; "
            "not an exact intraday line or roster snapshot replay."
        ),
    }
    write_json_atomic(root / "manifest.json", manifest)
    overall = comparison.deltas.loc[comparison.deltas["cut_type"] == "overall"]
    lines = [
        "# Expected-Points Backtest Comparison", "",
        f"League: {comparison.baseline.league.value.upper()}", "",
        (
            "> Weekly historical approximation using currently available feed values; "
            "not exact intraday replay."
        ), "",
        "| Metric | Baseline | Candidate | Improvement | 95% CI | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in overall.itertuples(index=False):
        interval = _display_interval(
            (row.improvement_ci_low, row.improvement_ci_high)
        )
        lines.append(
            f"| {row.metric} | {_display_metric(row.baseline, digits=4)} | "
            f"{_display_metric(row.candidate, digits=4)} | "
            f"{_display_metric(row.improvement, digits=4)} | {interval} | {row.status} |"
        )
    write_text_atomic(root / "report.md", "\n".join(lines) + "\n")
    release_metrics = overall.loc[
        overall["status"].isin(["improvement", "regression", "inconclusive"])
    ]
    release_lines = [
        "## Evaluation", "",
        "| Metric | Baseline | Candidate | Improvement | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in release_metrics.itertuples(index=False):
        release_lines.append(
            f"| {row.metric} | {_display_metric(row.baseline, digits=4)} | "
            f"{_display_metric(row.candidate, digits=4)} | "
            f"{_display_metric(row.improvement, digits=4)} | {row.status} |"
        )
    write_text_atomic(
        root / "release_evaluation.md",
        "\n".join(release_lines) + "\n",
    )
    return root


def write_run_artifacts(run: BacktestRun, output_dir: str | Path) -> Path:
    from .backtesting import dependency_fingerprint

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(root / "predictions.parquet", run.predictions)
    write_json_atomic(root / "summary.json", run.summary.to_dict(orient="records"))
    write_text_atomic(root / "report.md", render_run_report(run))
    write_json_atomic(root / "manifest.json", {
        "artifact_type": "expected_points_backtest_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "league": run.league.value,
        "recipe_name": run.recipe_name,
        "recipe_version": run.recipe_version,
        "recipe_fingerprint": run.recipe_fingerprint,
        "profile": run.profile,
        "seasons": run.seasons,
        "cache_hits": run.cache_hits,
        "trained_cutoffs": run.trained_cutoffs,
        "elapsed_seconds": run.elapsed_seconds,
        "cutoff_timings": run.cutoff_timings,
        "source_fingerprint": run.source_fingerprint,
        "json_columns": json_object_columns(run.predictions),
        "git_identity": _git_identity(),
        "python": platform.python_version(),
        "dependency_fingerprint": dependency_fingerprint(),
        "historical_fidelity": (
            "Weekly pre-first-kickoff replay using currently available historical feeds; "
            "not an exact intraday line or roster snapshot replay."
        ),
    })
    return root


def load_backtest_run(path: str | Path) -> BacktestRun:
    from .backtesting import BacktestRun

    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "expected_points_backtest_run":
        raise ValueError(f"Not an expected-points run artifact: {root}")
    return BacktestRun(
        league=ExpectedPointsLeague(manifest["league"]),
        recipe_name=manifest["recipe_name"],
        recipe_version=manifest["recipe_version"],
        recipe_fingerprint=manifest["recipe_fingerprint"],
        profile=manifest["profile"],
        seasons=tuple(int(value) for value in manifest["seasons"]),
        predictions=restore_json_columns(
            pd.read_parquet(root / "predictions.parquet"),
            manifest.get("json_columns", []),
        ),
        summary=pd.read_json(root / "summary.json", orient="records"),
        cache_hits=int(manifest.get("cache_hits", 0)),
        trained_cutoffs=int(manifest.get("trained_cutoffs", 0)),
        source_fingerprint=manifest["source_fingerprint"],
        elapsed_seconds=float(manifest.get("elapsed_seconds", 0.0)),
        cutoff_timings=tuple(manifest.get("cutoff_timings", ())),
    )


def save_backtest_frame(
    frame: pd.DataFrame,
    league: ExpectedPointsLeague | str,
    *,
    root: str | Path = Path(".backtests/expected_points"),
) -> Path:
    from .backtesting import frame_fingerprint

    parsed = league if isinstance(league, ExpectedPointsLeague) else ExpectedPointsLeague(league)
    destination = Path(root) / "frames" / parsed.value / "latest.parquet"
    write_parquet_atomic(destination, frame)
    write_json_atomic(destination.with_suffix(".json"), {
        "league": parsed.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": frame_fingerprint(frame),
        "rows": len(frame),
        "columns": list(frame.columns),
        "json_columns": json_object_columns(frame),
    })
    return destination


def load_backtest_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_parquet(source)
    metadata_path = source.with_suffix(".json")
    if not metadata_path.exists():
        return frame
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return restore_json_columns(frame, metadata.get("json_columns", []))


__all__ = [
    "json_object_columns",
    "load_backtest_frame",
    "load_backtest_run",
    "parquet_safe_frame",
    "render_run_report",
    "restore_json_columns",
    "save_backtest_frame",
    "write_comparison_artifacts",
    "write_json_atomic",
    "write_parquet_atomic",
    "write_run_artifacts",
    "write_text_atomic",
]
