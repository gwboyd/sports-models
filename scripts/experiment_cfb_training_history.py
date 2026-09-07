"""Compare explicit CFB training windows while freezing every existing input.

This advanced, read-only experiment requires saved frame bundles and a completed
reference run. It isolates training-history changes by preserving all existing
features, beyond the normal CLI's shared-outcome/market compatibility checks. It never fetches data or writes operational state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_patterns.expected_points.backtesting import (
    BacktestSpec, compare_backtest_runs, load_backtest_frame, load_backtest_run,
    recipe_code_identity, run_walk_forward, save_backtest_frame,
    source_bundle_fingerprint,
)
from src.model_patterns.expected_points.backtest_artifacts import (
    write_comparison_artifacts, write_json_atomic, write_run_artifacts,
)
from src.model_patterns.expected_points.backtest_workflow import (
    preflight_artifact, preflight_frames,
)
from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football.cfb.expected_points.recipe import CFBExpectedPointsRecipe
from src.sports.football.kickoff import parse_eastern_kickoffs


def validate_history_extension(
    reference: pd.DataFrame,
    extended: pd.DataFrame,
    spec: BacktestSpec,
    *,
    excluded_seasons: tuple[int, ...] = (2020,),
) -> dict:
    """Allow only additional earlier rows; reject changes to any retained value."""
    if extended.game_id.isna().any() or extended.game_id.astype(str).duplicated().any():
        raise ValueError('Extended history requires unique, non-null game IDs')
    if extended.season.isin(excluded_seasons).any():
        raise ValueError('Excluded seasons remain in extended training history')
    if set(reference.columns) != set(extended.columns):
        raise ValueError('Training-history experiments require identical feature schemas')
    left = reference.assign(game_id=reference.game_id.astype(str)).set_index('game_id')
    right = extended.assign(game_id=extended.game_id.astype(str)).set_index('game_id')
    if len(left.index.difference(right.index)):
        raise ValueError('Extended history removed reference games')
    try:
        pd.testing.assert_frame_equal(
            left.sort_index(), right.loc[left.index, left.columns].sort_index(),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError('Extended history changed existing feature/source values') from exc
    extra = extended.loc[~extended.game_id.astype(str).isin(left.index)]
    if extra.empty:
        raise ValueError('Training-history experiment added no earlier games')
    dates = parse_eastern_kickoffs(extra.date_time)
    if (extra.season.ge(reference.season.min()).any() or dates.isna().any()
            or dates.ge(parse_eastern_kickoffs(reference.date_time).min()).any()):
        raise ValueError('Additional training games must precede the entire reference frame')
    if not spec.seasons or min(spec.seasons) <= int(reference.season.min()):
        raise ValueError('Explicit evaluation seasons must follow reference warmup history')
    ref_eval = reference.loc[reference.season.isin(spec.seasons)].copy()
    # Keep the ordinary universe check unchanged, applying it to the explicitly
    # fixed evaluation population rather than deliberately different history.
    preflight_frames(reference, extended.loc[extended.season >= reference.season.min()], spec)
    return {
        'design': 'additional earlier training rows; all existing values frozen',
        'excluded_seasons': excluded_seasons,
        'reference_source_fingerprint': source_bundle_fingerprint(reference),
        'candidate_source_fingerprint': source_bundle_fingerprint(extended),
        'evaluation_source_fingerprint': source_bundle_fingerprint(ref_eval),
        'reference_rows': len(reference), 'candidate_rows': len(extended),
        'added_rows_by_season': extra.groupby('season').size().to_dict(),
        'evaluation_rows': len(ref_eval),
    }


def compare_training_windows(reference_run, candidate_run, design: dict):
    """Compare only a validated fixed population, preserving original run IDs.

    The original run artifacts retain their full-history source fingerprints.
    Temporary comparison views identify the evaluation population explicitly;
    they must never be reused as fitting caches or baseline source artifacts.
    """
    if (reference_run.source_fingerprint != design['reference_source_fingerprint']
            or candidate_run.source_fingerprint != design['candidate_source_fingerprint']):
        raise ValueError('Run source fingerprints do not match the validated history design')
    evaluation_id = design['evaluation_source_fingerprint']
    return compare_backtest_runs(
        replace(reference_run, source_fingerprint=evaluation_id),
        replace(candidate_run, source_fingerprint=evaluation_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-frame', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--history-frame', type=Path, required=True,
                        help='Same reference rows plus explicitly prepared earlier games')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--starts', nargs='+', type=int, default=[2018, 2015, 2013])
    parser.add_argument('--exclude-seasons', nargs='*', type=int, default=[2020])
    parser.add_argument('--seasons', default='2023,2024,2025')
    parser.add_argument('--profile', choices=['quick', 'standard', 'full'], default='standard')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.output.mkdir(parents=True, exist_ok=False)
    reference = load_backtest_frame(args.reference_frame)
    history = load_backtest_frame(args.history_frame)
    reference_run = load_backtest_run(args.reference_run)
    spec = BacktestSpec(profile=args.profile, seasons=tuple(map(int, args.seasons.split(','))))
    preflight_artifact(reference_run, reference, spec, 'cfb')
    recipe = CFBExpectedPointsRecipe()
    identity = recipe_code_identity(ExpectedPointsLeague.CFB)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    request = {**vars(args), 'code_identity': identity, 'script_sha256': script_hash,
               'input_bundles': {str(p): json.loads(p.with_suffix('.json').read_text())
                                for p in (args.reference_frame, args.history_frame)}}
    write_json_atomic(args.output/'request.json', request)
    variants = []
    for start in args.starts:
        frame = history.loc[(history.season >= start)
                            & ~history.season.isin(args.exclude_seasons)].copy()
        if start not in set(frame.season):
            raise ValueError(f'Requested start season {start} has no training rows')
        design = validate_history_extension(
            reference, frame, spec, excluded_seasons=tuple(args.exclude_seasons),
        )
        destination = args.output/f'start-{start}'
        save_backtest_frame(frame, ExpectedPointsLeague.CFB,
                            destination=destination/'frame.parquet',
                            provenance={**request, 'start': start, 'design': design})
        write_json_atomic(destination/'design.json', design)
        variants.append((frame, destination, design))
    # All variants preflight before any fits. Existing completed runs remain
    # available if a later variant fails; never overlap an automatic retry.
    for frame, destination, design in variants:
        logging.info('START %s: %s rows', destination.name, len(frame))
        candidate = run_walk_forward(frame, recipe, spec)
        write_run_artifacts(candidate, destination/'candidate')
        comparison = compare_training_windows(reference_run, candidate, design)
        write_comparison_artifacts(comparison, output_dir=destination)
        if (recipe_code_identity(ExpectedPointsLeague.CFB) != identity
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash):
            raise RuntimeError('Code changed during experiment; do not use these results')
        logging.info('COMPLETE %s', destination)


if __name__ == '__main__':
    main()
