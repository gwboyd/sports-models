"""Read-only opponent-adjustment experiments on explicitly frozen source snapshots.

The normal evaluation entry point remains ``make backtest-expected-points``.
This advanced runner reuses the shared walk-forward fitter and paired metrics,
with a fixed deployed frame/run as the evaluation population. Earlier source
history affects efficiency features only; it does not extend score-training rows.
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
    BacktestSpec, _periods, compare_backtest_runs, load_backtest_frame,
    load_backtest_run, recipe_code_identity, run_walk_forward,
    save_backtest_frame, select_backtest_seasons, summarize_predictions,
)
from src.model_patterns.expected_points.backtest_artifacts import (
    write_comparison_artifacts, write_json_atomic, write_run_artifacts,
)
from src.model_patterns.expected_points.backtest_workflow import preflight_frames
from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football.transforms import OpponentAdjustmentConfig
from src.sports.football.cfb.expected_points.features import build_opponent_adjusted_advanced_stats
from src.sports.football.cfb.expected_points.recipe import CFBExpectedPointsRecipe
from src.sports.football.nfl.expected_points.features import build_nfl_opponent_adjusted_metrics
from src.sports.football.nfl.expected_points.recipe import NFLExpectedPointsRecipe


def replace_team_metrics(base: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Join newly calculated team features without changing the fixed population."""
    output = base.copy()
    keys = ['game_id', 'team']
    values = [column for column in metrics if 'ewma' in column]
    metric_frame = metrics[keys + values].copy()
    metric_frame['game_id'] = metric_frame.game_id.astype(str)
    original_ids = output.game_id.copy()
    output['game_id'] = output.game_id.astype(str)
    for venue in ('home', 'away'):
        renamed = metric_frame.rename(columns={'team': f'{venue}_team', **{c: f'{c}_{venue}' for c in values}})
        output = output.drop(columns=[f'{c}_{venue}' for c in values], errors='ignore').merge(
            renamed, on=['game_id', f'{venue}_team'], how='left', validate='one_to_one', sort=False,
        )
    output['game_id'] = original_ids.to_numpy()
    return output


def build_frame(args, base: pd.DataFrame) -> pd.DataFrame:
    source = Path(args.sources)
    config = OpponentAdjustmentConfig(
        ridge_alpha=args.alpha, season_carryover=args.carryover,
        adjustment_strength=args.strength, fcs_policy=args.fcs_policy,
        excluded_seasons=tuple(args.exclude_seasons),
        smoothing_span=args.span,
    )
    schedule = pd.read_parquet(source / 'schedule.parquet')
    if args.history_start is not None:
        schedule = schedule.loc[schedule.season >= args.history_start].copy()
    if args.league == 'cfb':
        advanced = pd.read_parquet(source / 'advanced_game_stats.parquet')
        advanced = advanced.loc[advanced.game_id.isin(schedule.game_id)]
        metrics = build_opponent_adjusted_advanced_stats(
            advanced, schedule, config=config, metrics=tuple(args.metrics.split(',')),
        )
    else:
        pbp = pd.read_parquet(source / 'pbp.parquet')
        epa, success = build_nfl_opponent_adjusted_metrics(pbp, schedule, config=config)
        metrics = epa.merge(success, on=['game_id', 'season', 'week', 'team'], validate='one_to_one')
    return replace_team_metrics(base, metrics)


def blend_metric_frames(raw: pd.DataFrame, adjusted: pd.DataFrame, strength: float) -> pd.DataFrame:
    """Exploit linear correction/EWMA to screen strengths without refitting ratings.

    The two inputs must use identical observations, rating settings, and smoothing,
    differing only in strength zero versus one. Their manifests remain required
    experiment evidence. Unchanged values are retained exactly, including NaNs.
    """
    if not 0 <= strength <= 1:
        raise ValueError('strength must be between zero and one')
    if not raw.game_id.equals(adjusted.game_id) or list(raw.columns) != list(adjusted.columns):
        raise ValueError('Blend endpoints must have identical row and column order')
    result = raw.copy()
    for column in raw:
        equal = raw[column].eq(adjusted[column]) | (raw[column].isna() & adjusted[column].isna())
        if equal.all():
            continue
        if 'ewma' not in column:
            raise ValueError(f'Blend endpoints differ outside efficiency metrics: {column}')
        result.loc[~equal, column] = (
            raw.loc[~equal, column] + strength * (adjusted.loc[~equal, column] - raw.loc[~equal, column])
        )
    return result


def validate_blend_provenance(raw_path: Path, adjusted_path: Path) -> None:
    """Reject endpoints that change anything besides the correction fraction."""
    raw = json.loads(raw_path.with_suffix('.json').read_text()).get('provenance', {})
    adjusted = json.loads(adjusted_path.with_suffix('.json').read_text()).get('provenance', {})
    if raw.get('strength') != 0 or adjusted.get('strength') != 1:
        raise ValueError('Blend endpoints must record strengths zero and one')
    if raw.get('span') != adjusted.get('span'):
        raise ValueError('Blend endpoint provenance differs: span')
    for key in ('league', 'alpha', 'carryover', 'fcs_policy', 'history_start', 'exclude_seasons',
                'metrics', 'source_sha256', 'code_identity'):
        if key not in raw or raw[key] != adjusted.get(key):
            raise ValueError(f'Blend endpoint provenance differs: {key}')


def subset_baseline(run, frame: pd.DataFrame, spec: BacktestSpec):
    """Use exactly the requested cutoffs from an already completed standard run."""
    seasons = select_backtest_seasons(frame, spec)
    periods = set(_periods(frame, seasons, spec))
    predictions = run.predictions.loc[
        [(int(s), w) in periods for s, w in zip(run.predictions.season, run.predictions.week)]
    ].copy()
    expected = frame.loc[[(int(s), w) in periods for s, w in zip(frame.season, frame.week)]]
    if set(predictions.game_id.astype(str)) != set(expected.game_id.astype(str)):
        raise ValueError('Saved baseline does not cover every requested evaluated game')
    return replace(run, profile=spec.profile, seasons=seasons, predictions=predictions,
                   summary=summarize_predictions(predictions), trained_cutoffs=0,
                   cache_hits=len(periods), cutoff_timings=(), elapsed_seconds=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--league', choices=['nfl', 'cfb'], required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True, help='Completed deployed comparison directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['quick', 'standard'], default='quick')
    p.add_argument('--seasons', default='2023,2024', help='Screen on earlier seasons; reserve the latest for confirmation')
    p.add_argument('--alpha', type=float, default=5.0)
    p.add_argument('--carryover', type=float, default=0.5)
    p.add_argument('--strength', type=float, default=1.0)
    p.add_argument('--span', type=int, help='Override the smoothing span; dynamic spans still grow with week')
    p.add_argument('--fcs-policy', choices=['partial_pool', 'pooled'], default='partial_pool')
    p.add_argument('--history-start', type=int)
    p.add_argument('--exclude-seasons', nargs='*', type=int, default=[])
    p.add_argument('--metrics', default='explosiveness')
    p.add_argument('--prepared-frame', type=Path, help='Explicit frame produced by this experiment tooling')
    p.add_argument('--raw-frame', type=Path, help='Strength-zero endpoint for linear blending')
    p.add_argument('--adjusted-frame', type=Path, help='Matching strength-one endpoint')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.output.mkdir(parents=True, exist_ok=False)
    league = ExpectedPointsLeague(args.league)
    identity = recipe_code_identity(league)
    request = {**vars(args), 'code_identity': identity, 'source_sha256': {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.sources.glob('*.parquet'))
    }}
    request['input_frames'] = {
        str(path): json.loads(path.with_suffix('.json').read_text())
        for path in (args.prepared_frame, args.raw_frame, args.adjusted_frame) if path is not None
    }
    write_json_atomic(args.output/'request.json', request)
    base = load_backtest_frame(args.baseline/'frames/baseline.parquet')
    if bool(args.raw_frame) != bool(args.adjusted_frame) or (args.prepared_frame and args.raw_frame):
        raise ValueError('Use a prepared frame OR both blend endpoints')
    if args.raw_frame:
        validate_blend_provenance(args.raw_frame, args.adjusted_frame)
        frame = blend_metric_frames(load_backtest_frame(args.raw_frame), load_backtest_frame(args.adjusted_frame), args.strength)
    else:
        frame = load_backtest_frame(args.prepared_frame) if args.prepared_frame else build_frame(args, base)
    spec = BacktestSpec(profile=args.profile, seasons=tuple(map(int,args.seasons.split(','))))
    preflight_frames(base, frame, spec)
    save_backtest_frame(frame, league, destination=args.output/'frame.parquet', provenance=request)
    if args.prepare_only:
        print('PREPARED', args.output, flush=True)
        return
    recipe = (CFBExpectedPointsRecipe(efficiency_metrics=tuple(args.metrics.split(',')))
              if args.league == 'cfb' else NFLExpectedPointsRecipe())
    candidate = run_walk_forward(frame, recipe, spec)
    write_run_artifacts(candidate, args.output/'candidate')
    baseline = subset_baseline(load_backtest_run(args.baseline/'baseline'), base, spec)
    comparison = compare_backtest_runs(baseline, candidate)
    write_comparison_artifacts(comparison, output_dir=args.output)
    if recipe_code_identity(league) != identity:
        raise RuntimeError('Recipe code changed during experiment; do not use this result')
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
