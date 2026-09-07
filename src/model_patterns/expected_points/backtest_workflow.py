"""Preparation and fail-fast checks for read-only historical comparisons."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from .backtesting import (
    BacktestComparison,
    BacktestRun,
    BacktestSpec,
    _periods,
    select_backtest_seasons,
    source_bundle_fingerprint,
    source_bundle_frame,
)
from src.sports.football.kickoff import parse_eastern_kickoffs


def historical_frame(frame: pd.DataFrame, through_season: int | None) -> pd.DataFrame:
    """Apply an explicit upper bound without dropping earlier training history."""
    if through_season is None:
        return frame.copy()
    result = frame.loc[pd.to_numeric(frame["season"], errors="raise") <= through_season].copy()
    if result.empty:
        raise ValueError(f"No frame rows at or before season {through_season}")
    return result


def _source_difference(baseline: pd.DataFrame, candidate: pd.DataFrame) -> str:
    left, right = source_bundle_frame(baseline), source_bundle_frame(candidate)
    left = left.assign(game_id=left["game_id"].astype(str)).set_index("game_id")
    right = right.assign(game_id=right["game_id"].astype(str)).set_index("game_id")
    only_left, only_right = left.index.difference(right.index), right.index.difference(left.index)
    common = left.index.intersection(right.index)
    differences = {}
    for column in left.columns.intersection(right.columns):
        a, b = left.loc[common, column], right.loc[common, column]
        equal = (a.eq(b) | (a.isna() & b.isna())).fillna(False)
        if not equal.all():
            differences[column] = {"count": int((~equal).sum()), "game_ids": common[~equal].tolist()[:5]}
    return json.dumps({
        "baseline_only": {"count": len(only_left), "game_ids": only_left.tolist()[:5]},
        "candidate_only": {"count": len(only_right), "game_ids": only_right.tolist()[:5]},
        "different_columns": differences,
        "baseline_schema": {col: str(dtype) for col, dtype in left.dtypes.items()},
        "candidate_schema": {col: str(dtype) for col, dtype in right.dtypes.items()},
    }, sort_keys=True)


def preflight_frames(baseline: pd.DataFrame, candidate: pd.DataFrame, spec: BacktestSpec) -> dict:
    """Reject incompatible inputs before either expensive walk-forward fit."""
    required = {"game_id", "season", "week", "date_time", "home_team", "away_team", "home_score", "away_score"}
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} frame missing source columns: {sorted(missing)}")
        if frame["game_id"].isna().any() or frame["game_id"].astype(str).duplicated().any():
            raise ValueError(f"{label} frame requires unique, non-null game IDs")
        for market in ("spread", "total"):
            if not {f"{market}_reference_line", f"{market}_line"}.intersection(frame.columns):
                raise ValueError(f"{label} frame missing {market} market inputs")
    seasons = select_backtest_seasons(baseline, spec)
    if seasons != select_backtest_seasons(candidate, spec):
        raise ValueError("Frame preflight failed: evaluated seasons differ")
    first_season = min(seasons)
    left_eval = baseline.loc[baseline.season >= first_season]
    right_eval = candidate.loc[candidate.season >= first_season]
    left_eval_hash = source_bundle_fingerprint(left_eval)
    right_eval_hash = source_bundle_fingerprint(right_eval)
    if left_eval_hash != right_eval_hash:
        raise ValueError(
            "Frame preflight failed: game/outcome/market inputs differ. "
            + _source_difference(left_eval, right_eval)
            + ". Rebuild from matching inputs, or explicitly bound BOTH frames with --through-season. "
            "Do not intersect game IDs or replace feature columns to force compatibility."
        )
    left_ids, right_ids = set(baseline.game_id.astype(str)), set(candidate.game_id.astype(str))
    # Audit all shared history too: an older correction must not silently apply
    # to only one recipe. This is validation, never intersection of model inputs.
    left_common = baseline.loc[baseline.game_id.astype(str).isin(right_ids)]
    right_common = candidate.loc[candidate.game_id.astype(str).isin(left_ids)]
    if source_bundle_fingerprint(left_common) != source_bundle_fingerprint(right_common):
        raise ValueError("Frame preflight failed: shared historical game/outcome/market inputs differ. "
                         + _source_difference(left_common, right_common))
    first_kickoff = parse_eastern_kickoffs(left_eval.date_time).min()
    for label, data, other_ids in (("baseline", baseline, right_ids), ("candidate", candidate, left_ids)):
        unique_history = data.loc[~data.game_id.astype(str).isin(other_ids)]
        kickoffs = parse_eastern_kickoffs(unique_history.date_time)
        if (unique_history.season.ge(first_season).any() or kickoffs.isna().any()
                or kickoffs.ge(first_kickoff).any()):
            raise ValueError(f"Frame preflight failed: {label}-only training rows must precede evaluation")
    left_hash, right_hash = source_bundle_fingerprint(baseline), source_bundle_fingerprint(candidate)
    return {
        "status": "passed", "source_fingerprint": left_eval_hash,
        "baseline_source_fingerprint": left_hash, "candidate_source_fingerprint": right_hash,
        "training_history_differs": left_hash != right_hash,
        "baseline_rows": len(baseline), "candidate_rows": len(candidate),
        "evaluation_source_rows": len(left_eval),
        "baseline_training_seasons": baseline.loc[baseline.season < first_season].groupby('season').size().to_dict(),
        "candidate_training_seasons": candidate.loc[candidate.season < first_season].groupby('season').size().to_dict(),
        "rows": len(baseline), "profile": spec.profile, "seasons": list(seasons),
        "cutoffs": _periods(baseline, seasons, spec),
        "baseline_feature_columns": len(baseline.columns), "candidate_feature_columns": len(candidate.columns),
    }


def compare_prepared_runs(
    baseline: BacktestRun, candidate: BacktestRun,
    baseline_frame: pd.DataFrame, candidate_frame: pd.DataFrame, spec: BacktestSpec,
    *, bootstrap_samples: int = 2000,
) -> BacktestComparison:
    """Pair verified runs on fixed evaluation inputs, allowing earlier history changes.

    Original saved runs keep full-history identities for cache/artifact validation.
    Only the comparison views share the separately verified evaluation identity.
    """
    from .backtesting import compare_backtest_runs

    audit = preflight_frames(baseline_frame, candidate_frame, spec)
    preflight_artifact(baseline, baseline_frame, spec, baseline.league.value)
    preflight_artifact(candidate, candidate_frame, spec, baseline.league.value)
    return compare_backtest_runs(
        replace(baseline, source_fingerprint=audit['source_fingerprint']),
        replace(candidate, source_fingerprint=audit['source_fingerprint']),
        bootstrap_samples=bootstrap_samples,
    )


def preflight_artifact(run: BacktestRun, frame: pd.DataFrame, spec: BacktestSpec, league: str) -> None:
    """Check saved evidence against this request before fitting its counterpart."""
    seasons = select_backtest_seasons(frame, spec)
    periods = set(_periods(frame, seasons, spec))
    selected = frame.loc[[(int(s), w) in periods for s, w in zip(frame.season, frame.week)]]
    keys = lambda df: set(zip(df.season.astype(int), df.week.astype(str), df.game_id.astype(str)))
    if (run.league.value != league or run.profile != spec.profile or run.seasons != seasons
            or run.source_fingerprint != source_bundle_fingerprint(frame)
            or keys(run.predictions) != keys(selected)):
        raise ValueError("Artifact preflight failed: league, profile, seasons, source, or evaluated games differ")


def preparation_notebook(source: Path, *, export_root: Path, current_year: int, current_week: int) -> object:
    """Trim at the supported export boundary; never execute training/persistence cells.

    The exact legacy marker supports deployed protocol-capable notebooks which
    predate this command. Fail closed when the boundary is missing or ambiguous.
    """
    import nbformat

    notebook = nbformat.read(source, as_version=4)
    boundaries = [i for i, cell in enumerate(notebook.cells)
                  if cell.cell_type == "code" and cell.source.startswith("backtest_frame_path = None\n")]
    parameters = [i for i, cell in enumerate(notebook.cells)
                  if "parameters" in cell.metadata.get("tags", [])]
    if len(boundaries) != 1 or len(parameters) != 1 or parameters[0] >= boundaries[0]:
        raise ValueError(f"Notebook lacks an unambiguous pre-training frame-save boundary: {source}")
    notebook = copy.deepcopy(notebook)
    notebook.cells = notebook.cells[:boundaries[0]]
    settings = {
        "current_year": current_year, "current_week": current_week,
        "client_name": "notebook", "allow_non_aws_write": False,
        "strict_data_validation": True, "write_backtest_frame": False,
        "run_historical_backtest": False,
    }
    injected = nbformat.v4.new_code_cell(
        "import logging\nlogging.basicConfig(level=logging.INFO, format='%(message)s')\n"
        + "\n".join(f"{key} = {value!r}" for key, value in settings.items())
    )
    notebook.cells.insert(parameters[0] + 1, injected)
    notebook.cells.append(nbformat.v4.new_code_cell(
        "from src.model_patterns.expected_points.backtesting import save_backtest_frame\n"
        f"save_backtest_frame(df, expected_points_recipe.league, root={str(export_root)!r})\n"
    ))
    for cell in notebook.cells:
        # nbformat's new-cell helpers add IDs, but legacy 4.4 release notebooks
        # cannot contain them. Preserve the source schema and executable code.
        if notebook.nbformat_minor < 5:
            cell.pop("id", None)
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.validate(notebook)
    return notebook
