# Expected-Points Backtesting

This guide covers the shared NFL/CFB evaluation workflow for normal training health checks, historical single-recipe
runs, and baseline-versus-candidate comparisons. The historical runner is read-only: it never writes picks, results,
updates, or releases to Supabase and never saves fitted estimators.

## Choose the evaluation tier

- Every normal notebook/API train produces a cheap chronological score holdout plus an untouched confidence/lock
  health tail. Use it to catch obvious regressions on every run.
- `quick` evaluates the latest completed/evaluable season every fourth chronological week plus the final week. Use it
  for iteration and smoke testing.
- `standard` evaluates every week in the latest three completed/evaluable seasons. Use it for normal release evidence.
- `full` evaluates every week in the latest four completed/evaluable seasons and fails if sufficient prior history is
  unavailable.

Every selected historical week performs the complete score and confidence tuning/refit. Profiles reduce the number of
cutoffs; they do not make an individual cutoff less faithful.

## Prepare a model frame in a notebook

Use either NFL or CFB `expected_points/notebook.ipynb`:

1. Run the parameters, loading, preparation, feature, and model-frame assembly stages through visible `df` and
   `ep_config`.
2. Set `write_backtest_frame = True` and run **Save historical backtest frame (optional)**.
3. The cell writes `.backtests/expected_points/frames/<league>/latest.parquet` plus metadata JSON.

This save stage does not train the current model. To prepare data and immediately stop, run only through that cell.
The **Train model**, **Game-level inspection**, and **Historical baseline comparison** cells are independent stages.

Opponent-adjustment settings are part of recipe construction. When changing its ridge penalty, prior-season
carryover, CFB FCS-pooling policy, or rating snapshot cadence, regenerate the candidate frame before comparing it. The metric columns keep
their established names, so use a separately saved baseline frame whenever comparing a pre-adjustment release with
the adjusted candidate.

For a full interactive notebook run, set `run_historical_backtest = True`. Its comparison cell saves the current `df`
automatically when the frame-save cell was not enabled. Papermill/API runs leave both flags false and cannot run the
historical comparison. Notebook experiment objects are deep copies; permanent candidate methodology must still move
into `recipe.py` before it can be compared reproducibly.

## Compare a candidate with a baseline

The normal command is:

```sh
make backtest-expected-points \
  LEAGUE=nfl \
  PROFILE=standard \
  BASELINE=deployed \
  CANDIDATE=working-tree
```

Useful optional Make variables are:

```sh
make backtest-expected-points \
  LEAGUE=cfb \
  PROFILE=standard \
  SEASONS=2024,2025 \
  CADENCE=1 \
  BOOTSTRAP_SAMPLES=2000 \
  OUTPUT_DIR=.backtests/expected_points/comparisons/cfb/my-change
```

Released/version/SHA baselines cache by default. Mutable `working-tree` recipes neither read nor write cutoff caches
by default, ensuring each local comparison is a fresh fit. Set `CACHE_WORKING_TREE=1` when an expensive candidate run
needs interruption recovery or exact-state reuse:

```sh
make backtest-expected-points \
  LEAGUE=cfb \
  BASELINE=deployed \
  CANDIDATE=working-tree \
  CACHE_WORKING_TREE=1
```

Set `NO_CACHE=1` only when deliberately forcing both the baseline and candidate to retrain without reading or writing
cutoff caches.

When enabled, the mutable `working-tree` candidate uses one rolling cache namespace per league, so repeated local
edits do not accumulate fingerprint directories. Every cutoff inside that namespace still records and validates the
full current recipe, configuration, input, and kickoff fingerprints; a changed candidate retrains and atomically
replaces only the selected weekly entries. `deployed` and its corresponding `version:N.N` reference share the readable
`version-N.N` namespace keyed by that version's immutable registry SHA, with full metadata validation before reuse.
Explicit `sha:*` references retain independent fingerprinted namespaces. Cache lookup is local-only and is not part
of normal notebook/API training or production execution.

Baseline and candidate references accept:

- `deployed`: the latest live version and its release-registry `source_git_sha`;
- `version:N.N`: that live version's release-registry `source_git_sha`;
- `sha:<sha>`: an explicit commit;
- `artifact:<path>`: an existing single-run artifact;
- `working-tree`: current tracked and untracked recipe content, including dirty changes.

Historical Git references run in detached temporary worktrees. If dependency fingerprints differ, the runner creates
and reuses an environment under `.backtests/expected_points/environments/`. A reference from before the backtest
protocol existed must use a bootstrapped artifact instead. Routine pick runs and explicitly authorized local writes
never change baseline resolution. The bootstrap 1.0 registry rows originally have no SHA; the first verified
protocol-capable production deployment initializes that field exactly once, after which `deployed` and `version:1.0`
are directly runnable. Per-run pick-update SHAs are audit metadata only. A different canonical recipe SHA requires a
new model version and release row.

When feature construction differs between versions, save each frame and pass both paths:

```sh
make backtest-expected-points \
  LEAGUE=nfl \
  BASELINE=version:1.0 \
  CANDIDATE=working-tree \
  BASELINE_FRAME=/absolute/path/to/baseline.parquet \
  CANDIDATE_FRAME=/absolute/path/to/candidate.parquet
```

The frames may have different feature columns, but they must describe the same game, outcome, and market universe.

## Run one recipe without comparison

Use the CLI directly when only a historical record is needed:

```sh
.venv/bin/python scripts/backtest_expected_points.py run \
  --league nfl \
  --profile standard \
  --frame .backtests/expected_points/frames/nfl/latest.parquet \
  --recipe-version working-tree \
  --output-dir .backtests/expected_points/runs/nfl/current
```

Add `--cache-working-tree` to opt this single working-tree run into cutoff caching.

The command reports each cutoff as a cache hit or fresh train with elapsed time and ETA. A run artifact contains
`manifest.json`, `predictions.parquet`, `summary.json`, and `report.md`. The manifest records aggregate and per-cutoff
timing. The report includes overall/per-season records, season-week block-bootstrap intervals, confidence diagnostics,
and model-versus-market point-error benchmarks.

## Read results in Python or a notebook

Comparison cells bind `backtest_comparison` with ordinary DataFrames:

```python
comparison = load_backtest_comparison(output_dir)
comparison.deltas.query("cut_type == 'overall'")
comparison.candidate_predictions.query("season == 2025 and week == 7")
comparison.lock_changes.query("spread_lock_change != 'neither'")
```

For a single run:

```python
from src.model_patterns.expected_points.backtesting import load_backtest_run

run = load_backtest_run(".backtests/expected_points/runs/nfl/current")
run.summary.query("cut_type in ['overall', 'season', 'season_phase']")
run.predictions.query("game_id == 'example-game-id'")
```

Pushes are counted and excluded from win-rate denominators. A win rate with zero decisions is `null`/`N/A`, never
zero percent. `*_auc` near `0.500` means the confidence model is not ranking wins better than chance.
`*_lock_calibration_gap` is mean stated lock probability minus observed lock win rate. Positive
`*_mae_advantage_vs_market` means the score model beat the market-implied baseline; negative means the market baseline
had lower MAE.

## Cache behavior and reproducibility

All data and artifacts are local and Git-ignored under `.backtests/expected_points/`. `latest.parquet` is intentionally
the only notebook frame pointer; it can be recreated from the notebook and is not a permanent release artifact.

Each cutoff cache key covers recipe identity, protocol/configuration, dependencies, cutoff, and all frame inputs
available through that cutoff. A historical correction changes that cutoff and every later expanding-history fit.
Code/configuration changes use another recipe fingerprint. Cache deletion is always safe because recipes are unfitted
code and every model is retrained from its saved source frame.

When working-tree caching is explicitly enabled, a dirty run is resumable because dirty and untracked Python files
are discovered recursively throughout the shared expected-points package, the selected league package, and shared
football transforms. Their contents participate in the fingerprint, so adding a helper does not require maintaining
a filename allowlist. Release evidence should name a clean deployed Git SHA. Checked-in notebooks must have no
outputs or execution counts.

## Fidelity limits

Each simulated week freezes immediately before its earliest kickoff and predicts the full weekly slate once. The
source frame is loaded once; cutoffs filter that in-memory frame rather than calling APIs weekly. Historical feeds can
contain corrected or final line/roster values, so reports describe the result as a weekly approximation, not an exact
intraday replay. Increasing cutoff frequency does not fix that limitation without point-in-time source snapshots.
