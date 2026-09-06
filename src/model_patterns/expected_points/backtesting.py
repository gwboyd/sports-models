"""Read-only weekly walk-forward backtesting with optional cutoff caching.

Released versions use readable rolling cache namespaces, while explicit
immutable SHAs keep fingerprinted namespaces. Working-tree caching is opt-in.
Every cached cutoff still has to match the current recipe, configuration,
input, and kickoff before reuse.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.sports.football.kickoff import parse_eastern_kickoffs

from .backtest_artifacts import (
    _jsonable,
    load_backtest_frame,
    load_backtest_run,
    save_backtest_frame,
    write_comparison_artifacts,
    write_run_artifacts,
)
from .backtest_artifacts import (
    json_object_columns as _json_object_columns,
)
from .backtest_artifacts import (
    restore_json_columns as _restore_json_columns,
)
from .backtest_artifacts import (
    write_json_atomic as _write_json_atomic,
)
from .backtest_artifacts import (
    write_parquet_atomic as _write_parquet_atomic,
)
from .betting import calculate_wins
from .trainer import run_expected_points_at_cutoff
from .types import ExpectedPointsLeague, ExpectedPointsRecipe

LOGGER = logging.getLogger(__name__)

PROFILE_NAMES = ("quick", "standard", "full")
DEFAULT_CACHE_ROOT = Path(".backtests/expected_points")


@dataclass(frozen=True)
class BacktestSpec:
    profile: Literal["quick", "standard", "full"] = "standard"
    seasons: tuple[int, ...] = ()
    cadence: int | None = None
    cache_root: Path = DEFAULT_CACHE_ROOT
    use_cache: bool = True
    cache_working_tree: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.profile not in PROFILE_NAMES:
            raise ValueError(f"Unsupported backtest profile: {self.profile}")
        if self.cadence is not None and self.cadence < 1:
            raise ValueError("Backtest cadence must be at least one")


@dataclass(frozen=True)
class BacktestRun:
    league: ExpectedPointsLeague
    recipe_name: str
    recipe_version: str
    recipe_fingerprint: str
    profile: str
    seasons: tuple[int, ...]
    predictions: pd.DataFrame
    summary: pd.DataFrame
    cache_hits: int
    trained_cutoffs: int
    source_fingerprint: str
    elapsed_seconds: float = 0.0
    cutoff_timings: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class BacktestComparison:
    baseline: BacktestRun
    candidate: BacktestRun
    summary: pd.DataFrame
    deltas: pd.DataFrame
    lock_changes: pd.DataFrame


def _digest_json(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    sort_columns = [column for column in ("date_time", "season", "week", "game_id") if column in output]
    if sort_columns:
        output = output.sort_values(sort_columns, kind="stable")
    output = output.reindex(sorted(output.columns), axis=1)
    for column in output.select_dtypes(include=["object"]).columns:
        output[column] = output[column].map(
            lambda value: json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, (dict, list, tuple, set))
            else str(value)
        )
    return output.reset_index(drop=True)


def frame_fingerprint(frame: pd.DataFrame) -> str:
    stable = _stable_frame(frame)
    hashed = pd.util.hash_pandas_object(stable, index=True).values.tobytes()
    schema = json.dumps([(column, str(dtype)) for column, dtype in stable.dtypes.items()])
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8"))
    digest.update(hashed)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dependency_fingerprint(root: Path | None = None) -> str:
    """Identify the Python/runtime inputs that can change a fitted recipe."""
    root = root or _repository_root()
    digest = hashlib.sha256()
    digest.update(f"python:{platform.python_version()}".encode())
    for name in ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "uv.lock"):
        path = root / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def recipe_code_identity(league: ExpectedPointsLeague, root: Path | None = None) -> str:
    """Hash Git SHA plus relevant dirty files, including untracked recipe code."""
    root = root or _repository_root()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        digest = hashlib.sha256()
        digest.update(sha.encode("utf-8"))
        shared = root / "src/model_patterns/expected_points"
        league_root = root / "src/sports/football" / league.value
        # Discover package files recursively so a new dirty or untracked shared
        # helper cannot bypass invalidation merely because a filename allowlist
        # was not updated.
        paths = list(shared.rglob("*.py"))
        paths.extend(league_root.rglob("*.py"))
        paths.extend((root / "src/sports/football/transforms").rglob("*.py"))
        paths.append(root / "src/sports/football/kickoff.py")
        for path in sorted({path for path in paths if path.is_file()}):
            relative = path.relative_to(root)
            digest.update(str(relative).encode("utf-8"))
            digest.update(path.read_bytes())
        return f"{sha}:{digest.hexdigest()}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def recipe_fingerprint(recipe: ExpectedPointsRecipe, frame: pd.DataFrame, season: int, week: int) -> str:
    config = recipe.build_config(frame, season=season, week=week)
    try:
        source = inspect.getsource(recipe.__class__)
    except (OSError, TypeError):
        source = recipe.__class__.__qualname__
    return _digest_json({
        "league": recipe.league.value,
        "name": recipe.name,
        "version": recipe.version,
        "protocol_version": recipe.protocol_version,
        "code_identity": recipe_code_identity(recipe.league),
        "dependency_fingerprint": dependency_fingerprint(),
        "source": source,
        "config": config,
    })


def source_bundle_fingerprint(frame: pd.DataFrame) -> str:
    """Hash the shared evaluation universe, independent of recipe feature columns."""
    preferred = [
        "game_id", "season", "week", "date_time", "home_team", "away_team",
        "home_score", "away_score",
    ]
    columns = [column for column in preferred if column in frame]
    # CFB recipe changes may deliberately select a different executable quote,
    # so compare the shared consensus source rather than the derived execution line.
    for reference, fallback in (
        ("spread_reference_line", "spread_line"),
        ("total_reference_line", "total_line"),
    ):
        if reference in frame:
            columns.append(reference)
        elif fallback in frame:
            columns.append(fallback)
    return frame_fingerprint(frame[columns])


def _complete_seasons(frame: pd.DataFrame) -> list[int]:
    completed = []
    for season, group in frame.groupby("season", sort=True):
        scores = group[["home_score", "away_score"]].apply(pd.to_numeric, errors="coerce")
        if not group.empty and scores.notna().all(axis=1).all():
            completed.append(int(season))
    return completed


def select_backtest_seasons(frame: pd.DataFrame, spec: BacktestSpec) -> tuple[int, ...]:
    complete = _complete_seasons(frame)
    if not complete:
        raise ValueError("Backtesting requires at least one completed season")
    # The first complete season supplies prior history for confidence training.
    eligible = complete[1:]
    if spec.seasons:
        missing = sorted(set(spec.seasons) - set(eligible))
        if missing:
            raise ValueError(
                "Requested seasons lack a complete prior-history season: "
                + ", ".join(map(str, missing))
            )
        return tuple(sorted(spec.seasons))
    count = {"quick": 1, "standard": 3, "full": 4}[spec.profile]
    if len(eligible) < count:
        raise ValueError(
            f"Profile {spec.profile!r} requires {count} completed/evaluable seasons; "
            f"found {len(eligible)} ({', '.join(map(str, eligible)) or 'none'})"
        )
    return tuple(eligible[-count:])


def _periods(frame: pd.DataFrame, seasons: tuple[int, ...], spec: BacktestSpec) -> list[tuple[int, Any]]:
    selected = frame.loc[frame["season"].isin(seasons)].copy()
    selected["__kickoff"] = parse_eastern_kickoffs(selected["date_time"])
    periods = (
        selected.groupby(["season", "week"], dropna=False)["__kickoff"]
        .min()
        .sort_values(kind="stable")
        .index.tolist()
    )
    cadence = spec.cadence or (4 if spec.profile == "quick" else 1)
    sampled = periods[::cadence]
    if periods and periods[-1] not in sampled:
        sampled.append(periods[-1])
    return [(int(season), week) for season, week in sampled]


def _cutoff_cache_paths(
    spec: BacktestSpec,
    recipe: ExpectedPointsRecipe,
    fingerprint: str,
    season: int,
    week: Any,
) -> tuple[Path, Path]:
    safe_week = str(week).replace("/", "-")
    # Reuse one readable slot for the mutable working tree and one for each
    # immutable registered model version. Metadata validation still decides
    # whether changed inputs/configuration make a cutoff stale. Explicit SHA
    # runs remain independently addressable through fingerprinted namespaces.
    if recipe.version == "working-tree":
        namespace = "working-tree"
    elif recipe.version.startswith("version:"):
        namespace = f"version-{recipe.version.removeprefix('version:')}"
    else:
        namespace = fingerprint
    root = spec.cache_root / "cache" / recipe.league.value / namespace
    return root / f"{season}_{safe_week}.parquet", root / f"{season}_{safe_week}.json"


def _cutoff_input_fingerprint(frame: pd.DataFrame, target: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    kickoffs = parse_eastern_kickoffs(frame["date_time"])
    history = frame.loc[kickoffs < cutoff]
    return frame_fingerprint(pd.concat([history, target], ignore_index=True, sort=False))


def run_walk_forward(
    frame: pd.DataFrame,
    recipe: ExpectedPointsRecipe,
    spec: BacktestSpec | None = None,
) -> BacktestRun:
    """Train an unfitted recipe once at every selected historical weekly cutoff."""
    spec = spec or BacktestSpec()
    required = {"game_id", "season", "week", "date_time", "home_score", "away_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Backtest model frame is missing columns: {', '.join(missing)}")
    if not frame["game_id"].astype(str).is_unique:
        raise ValueError("Backtest model frame requires unique game_id values")

    seasons = select_backtest_seasons(frame, spec)
    periods = _periods(frame, seasons, spec)
    if not periods:
        raise ValueError("Backtest profile selected no historical periods")
    fingerprint = recipe_fingerprint(recipe, frame, periods[0][0], int(periods[0][1]))
    source_fingerprint = source_bundle_fingerprint(frame)
    cache_enabled = spec.use_cache and (
        recipe.version != "working-tree" or spec.cache_working_tree
    )
    predictions: list[pd.DataFrame] = []
    cache_hits = 0
    trained_cutoffs = 0
    cutoff_timings: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for cutoff_index, (season, week) in enumerate(periods, start=1):
        cutoff_started = time.perf_counter()
        target = frame.loc[(frame["season"] == season) & (frame["week"] == week)].copy()
        target_kickoffs = parse_eastern_kickoffs(target["date_time"])
        if target_kickoffs.isna().any():
            raise ValueError(f"Invalid kickoff in historical slate {season}/{week}")
        cutoff = target_kickoffs.min() - pd.Timedelta(nanoseconds=1)
        config = recipe.build_config(
            frame,
            season=season,
            week=int(week),
            prediction_now=cutoff,
        )
        prediction_path = metadata_path = None
        expected_metadata: dict[str, Any] = {}
        if cache_enabled:
            input_fingerprint = _cutoff_input_fingerprint(frame, target, cutoff)
            prediction_path, metadata_path = _cutoff_cache_paths(
                spec, recipe, fingerprint, season, week
            )
            expected_metadata = {
                "recipe_fingerprint": fingerprint,
                "config_fingerprint": _digest_json(config),
                "input_fingerprint": input_fingerprint,
                "season": season,
                "week": str(week),
                "cutoff": cutoff.isoformat(),
            }
        cached = None
        metadata: dict[str, Any] = {}
        if (
            cache_enabled
            and prediction_path is not None
            and metadata_path is not None
            and prediction_path.exists()
            and metadata_path.exists()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(metadata.get(key) == value for key, value in expected_metadata.items()):
                cached = _restore_json_columns(
                    pd.read_parquet(prediction_path),
                    metadata.get("json_columns", []),
                )
        if cached is not None:
            cache_hits += 1
            predictions.append(cached)
            elapsed = time.perf_counter() - cutoff_started
            cutoff_timings.append({
                "season": season,
                "week": str(week),
                "cache_hit": True,
                "elapsed_seconds": elapsed,
                "source_train_seconds": metadata.get("train_elapsed_seconds"),
                "rows": len(cached),
            })
            if spec.progress:
                total_elapsed = time.perf_counter() - run_started
                eta = total_elapsed / cutoff_index * (len(periods) - cutoff_index)
                LOGGER.info(
                    "[%s] cutoff %d/%d season=%s week=%s cache-hit rows=%d "
                    "elapsed=%.1fs eta=%.1fs",
                    recipe.league.value,
                    cutoff_index,
                    len(periods),
                    season,
                    week,
                    len(cached),
                    elapsed,
                    eta,
                )
            continue

        run = run_expected_points_at_cutoff(
            frame,
            config,
            cutoff=cutoff,
            prediction_frame=target,
        )
        graded = calculate_wins(run.plays)
        graded["backtest_cutoff"] = cutoff.isoformat()
        graded["recipe_name"] = recipe.name
        graded["recipe_version"] = recipe.version
        graded["recipe_fingerprint"] = fingerprint
        train_elapsed = time.perf_counter() - cutoff_started
        if cache_enabled and prediction_path is not None and metadata_path is not None:
            _write_parquet_atomic(prediction_path, graded)
            _write_json_atomic(metadata_path, {
                **expected_metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rows": len(graded),
                "train_elapsed_seconds": train_elapsed,
                "json_columns": _json_object_columns(graded),
                "health_metrics": run.metrics,
            })
        trained_cutoffs += 1
        predictions.append(graded)
        cutoff_timings.append({
            "season": season,
            "week": str(week),
            "cache_hit": False,
            "elapsed_seconds": train_elapsed,
            "source_train_seconds": train_elapsed,
            "rows": len(graded),
        })
        if spec.progress:
            total_elapsed = time.perf_counter() - run_started
            eta = total_elapsed / cutoff_index * (len(periods) - cutoff_index)
            LOGGER.info(
                "[%s] cutoff %d/%d season=%s week=%s trained rows=%d "
                "elapsed=%.1fs eta=%.1fs",
                recipe.league.value,
                cutoff_index,
                len(periods),
                season,
                week,
                len(graded),
                train_elapsed,
                eta,
            )

    combined = pd.concat(predictions, ignore_index=True, sort=False)
    combined = add_season_phase(combined)
    return BacktestRun(
        league=recipe.league,
        recipe_name=recipe.name,
        recipe_version=recipe.version,
        recipe_fingerprint=fingerprint,
        profile=spec.profile,
        seasons=seasons,
        predictions=combined,
        summary=summarize_predictions(combined),
        cache_hits=cache_hits,
        trained_cutoffs=trained_cutoffs,
        source_fingerprint=source_fingerprint,
        elapsed_seconds=time.perf_counter() - run_started,
        cutoff_timings=tuple(cutoff_timings),
    )


def add_season_phase(frame: pd.DataFrame) -> pd.DataFrame:
    from .backtest_metrics import add_season_phase as implementation

    return implementation(frame)


def summarize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    from .backtest_metrics import summarize_predictions as implementation

    return implementation(frame)


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    metrics: tuple[str, ...] | None = None,
    *,
    samples: int = 2000,
    seed: int = 31,
) -> dict[str, tuple[float, float]]:
    from .backtest_metrics import bootstrap_metric_intervals as implementation

    return implementation(frame, metrics, samples=samples, seed=seed)


def compare_backtest_runs(
    baseline: BacktestRun,
    candidate: BacktestRun,
    *,
    bootstrap_samples: int = 2000,
    random_seed: int = 31,
) -> BacktestComparison:
    from .backtest_metrics import compare_backtest_runs as implementation

    return implementation(
        baseline,
        candidate,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )


__all__ = [
    "BacktestComparison",
    "BacktestRun",
    "BacktestSpec",
    "add_season_phase",
    "bootstrap_metric_intervals",
    "compare_backtest_runs",
    "dependency_fingerprint",
    "frame_fingerprint",
    "load_backtest_frame",
    "load_backtest_run",
    "recipe_code_identity",
    "recipe_fingerprint",
    "run_walk_forward",
    "save_backtest_frame",
    "select_backtest_seasons",
    "source_bundle_fingerprint",
    "summarize_predictions",
    "write_comparison_artifacts",
    "write_run_artifacts",
]
