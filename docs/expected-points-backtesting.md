# Expected-Points Backtesting

## Evaluate working-tree changes against deployed

**The standard workflow is just to run the command from the repository root:**

```sh
make backtest-expected-points LEAGUE=nfl
```

For CFB:

```sh
make backtest-expected-points LEAGUE=cfb
```

With no `LEAGUE`, Make defaults to NFL. This is the normal evaluation path for **all prediction-affecting changes**:
adding, removing, or renaming features; changing feature calculations or opponent adjustment; changing training
windows, estimators, tuning, confidence inputs, or lock rules. Each version uses its own feature construction and
model recipe. Feature names and columns do not need to match. Both versions must predict comparable outcomes on the
same historical game/outcome/market universe. A change to that evaluation population requires an explicit evaluation
design; the runner never silently intersects games to manufacture a comparison. This shared runner currently supports
NFL and CFB expected-points models; another sport/model needs a recipe adapter.

The command automatically:

1. Resolves the deployed baseline to its immutable live release SHA and pins it for the entire job.
2. Prepares separate baseline and working-tree input tables using each version's code. Baseline preparation and
   training run in isolated Git checkouts; dependency differences use a cached environment for that version.
3. Checks frame integrity, code provenance, historical inputs, and evaluated seasons before fitting either version.
4. Runs **standard**: weekly walk-forward fitting across the latest three completed/evaluable seasons, with a prior
   season retained for warmup. Each cutoff uses the complete score and confidence tuning/refit.
5. Saves each run, paired performance results, prediction/lock changes, input bundles, executed preparation notebooks,
   logs, and the resolved request under a new `.backtests/expected_points/comparisons/<league>/<timestamp>/` directory.

There is no required notebook run, separate `prepare-frame` command, preflight command, quick run, or output-directory
selection. The terminal prints the job location, preparation progress, and per-cutoff progress/ETA. Start with
`report.md` for the comparison and `lock_changes.parquet` for changed decisions; `baseline/report.md` and
`candidate/report.md` describe each model separately. `request.json` preserves user options and
`resolved_request.json` records the effective dates, references, and input paths.

Before running, install the repository's development dependencies into `.venv` as described in the root README and
configure the root `.env`. `SUPABASE_DB_URL` is needed to resolve a deployed/version baseline; CFB preparation also
needs `CFBD_API_KEY`. Historical feed downloads require network access, and the referenced Git commit must exist
locally. Everything is read-only with respect to operational picks, results, updates, and releases.

Keep candidate code/settings unchanged while a job runs. Make permanent feature and training changes in the source
used by the notebook/recipe so preparation and fitting reproduce them. After another model change, run the same
command again; automatic preparation generates new frames rather than silently reusing `latest.parquet`.

### Default dates and cost

For automatic preparation, the history end defaults to the most recent conservatively closed football season:
from March onward, the previous calendar year; in January/February, two calendar years earlier. This common NFL/CFB
policy avoids declaring a season complete during a gap in currently quoted games. The resolved bound is logged.
Preparation loads through that year using notebook week-one context, exports before training, and retains all prior
history. Complete-season checks still apply; the runner fails if the selected profile lacks enough evaluable seasons.
To include a newly finished season before March, explicitly set `THROUGH_SEASON=<year>` after verifying completion.

**Standard runs directly; quick is optional.** Quick samples the latest evaluable season every fourth chronological
week plus the final week. It provides preliminary results sooner, but is not a prerequisite. With working-tree
cutoff caching off (the default), a later standard run repeats the overlapping candidate fits. Do not run both by
habit. Released baseline cutoffs can reuse compatible caches; newly prepared feed values can invalidate them.

## Optional controls

None of these options are required for the standard command above. CLI equivalents apply to
`.venv/bin/python scripts/backtest_expected_points.py compare --league nfl`, which has the same defaults as Make.

| Make option | CLI equivalent | When to use it |
|---|---|---|
| `LEAGUE=nfl` or `cfb` | `--league nfl` or `cfb` | Select the model. Make defaults to NFL; CLI requires a league. |
| `PROFILE=quick` | `--profile quick` | Optional smaller preliminary comparison; full tuning at fewer cutoffs. |
| `PROFILE=full` | `--profile full` | Evaluate four completed/evaluable seasons rather than standard's three. |
| `SEASONS=2023,2024,2025` | `--seasons 2023,2024,2025` | Explicit evaluation seasons; prior complete warmup history is still required. With fully automatic inputs, their maximum supplies the history end unless overridden. |
| `THROUGH_SEASON=2025` | `--through-season 2025` | Explicit inclusive upper history bound on both frames, retaining earlier training history. |
| `CADENCE=2` | `--cadence 2` | Sample every second chronological week; overrides profile cadence, not tuning fidelity. |
| `BASELINE=version:1.0` | `--baseline version:1.0` | Compare with a particular live registered release instead of `deployed`. |
| `BASELINE=sha:<sha>` | `--baseline sha:<sha>` | Compare with an explicit commit containing the backtest protocol. |
| `CANDIDATE=<reference>` | `--candidate <reference>` | Override `working-tree`; accepts the same references as the baseline. |
| `OUTPUT_DIR=<new-path>` | `--output-dir <new-path>` | Name the job directory instead of using an automatic timestamp. Must not already exist. |
| `CACHE_WORKING_TREE=1` | `--cache-working-tree` | Opt into resumable candidate cutoff caching. Automatic frame preparation still runs. |
| `NO_CACHE=1` | `--no-cache` | Disable reading/writing cutoff caches for both sides; does not discard explicit saved run artifacts. |
| `BOOTSTRAP_SAMPLES=2000` | `--bootstrap-samples 2000` | Change paired uncertainty resampling; default is 2,000 with fixed seed 31. |
| `PREFLIGHT_ONLY=1` | `--preflight-only` | Prepare/check inputs and stop before fitting; useful for investigating feeds or setup. |
| `BASELINE_FRAME=<path>` | `--baseline-frame <path>` | Reuse an explicitly selected baseline frame bundle instead of preparing it. |
| `CANDIDATE_FRAME=<path>` | `--candidate-frame <path>` | Reuse an explicitly selected candidate frame bundle instead of preparing it. |
| `FRAME=<path>` | `--frame <path>` | Advanced shared-frame override, only when both versions have identical feature construction. Side-specific paths take precedence. |
| `CURRENT_YEAR=<year>` | `--current-year <year>` | Advanced notebook loading context; defaults to the history end and must cover it. Usually leave unset. |
| `CURRENT_WEEK=<week>` | `--current-week <week>` | Advanced notebook loading context; default is 1. Does not select backtest weeks. |
| — | `--cache-root <path>` | Override the local cutoff/environment cache root. |

For example, an intentionally smaller preliminary run is:

```sh
make backtest-expected-points LEAGUE=cfb PROFILE=quick
```

To evaluate a specified history window with recoverable candidate cutoff fits:

```sh
make backtest-expected-points LEAGUE=nfl SEASONS=2023,2024,2025 CACHE_WORKING_TREE=1
```

The default command prepares both frames. If only one frame is supplied, it prepares the other side using that saved
frame's latest season as loading context, unless explicitly bounded. With saved inputs, the default preserves their
full source universe; `THROUGH_SEASON` can apply a deliberate shared upper bound. This supports recovery without
silently changing the population of an existing run. Saved inputs must still pass metadata and provenance checks.

## Saved results and recovery

Each comparison owns a fresh directory. It saves the actual bounded Parquet/JSON input bundles under `frames/`,
preparation notebooks and logs under `preparation/baseline/` and `preparation/candidate/`, `preflight.json`,
`execution.log`, and `status.json`. Finished fits are independently loadable run artifacts under `baseline/` and
`candidate/`. If the other fit or report fails, completed runs remain available, including when report rendering fails.

For example, reuse a completed baseline while retrying the candidate:

```sh
make backtest-expected-points LEAGUE=nfl \
  BASELINE=artifact:/absolute/path/to/previous-job/baseline \
  CANDIDATE_FRAME=/absolute/path/to/previous-job/frames/candidate.parquet
```

The runner automatically finds a comparison artifact's sibling input bundle. Older or standalone run artifacts need
an explicit matching `BASELINE_FRAME`/`CANDIDATE_FRAME`; predictions are not input features. Reusing the candidate
works the same way with `CANDIDATE=artifact:<path>`. Match the original profile, seasons, cadence, and source universe.
An artifact with different coverage fails preflight before new fitting. If candidate code changed, omit the stale
candidate frame so it is regenerated.

A hard-killed process may leave status `running`. Inspect the actual process/session and retained log before retrying;
do not infer failure from silence or launch overlapping retries for the same candidate. Use a new job directory for
every retry. Working-tree caches remain off unless explicitly enabled; a completed run artifact can be reused either
way. No cache or artifact option authorizes operational writes.

## Optional frame preparation and notebook inspection

A frame is the historical game table containing model inputs. Separate frame construction is an internal step of every
automatic comparison so each recipe uses its own feature logic. Standalone preparation remains available for debugging:

```sh
.venv/bin/python scripts/backtest_expected_points.py prepare-frame \
  --league nfl --recipe working-tree --current-year 2025 --current-week 1 \
  --through-season 2025 --output-dir .backtests/expected_points/frames/nfl/my-inspection
```

`--recipe` also accepts `deployed`, `version:N.N`, or `sha:<sha>`. It executes only through the notebook's pre-training
frame boundary, saves `frame.parquet`/`frame.json` plus the executed notebook and log, and stops before training,
inspection, comparison, or persistence. An absent/ambiguous boundary fails closed. Both the normal comparison and
standalone preparation use the selected source's settings; they record code and notebook identities.

The NFL/CFB notebooks remain interactive inspection surfaces. Set `write_backtest_frame=True` and run only through
**Save historical backtest frame (optional)** to export `df` to `frames/<league>/latest.parquet` without fitting.
Training, game inspection, and comparison are separate cells. `run_historical_backtest=True` opts into the notebook's
comparison cell, with `backtest_profile="standard"` by default; it supplies the current `df` as the candidate frame
and prepares the baseline separately. Papermill/API runs cannot launch that comparison. Experiment frames and
configuration must be copies, and lasting experiments must be reproducible in source.

Treat a saved frame as a Parquet **and JSON** bundle. Do not copy Parquet alone or reuse metadata after filtering.
The loader checks rows/columns and verifies new bundles' Parquet checksums. Legacy bundles without code provenance
emit a warning; regenerate them if their feature recipe is uncertain. For a named/filtered copy in Python:

```python
from src.model_patterns.expected_points.backtest_artifacts import load_backtest_frame, save_backtest_frame
from src.model_patterns.expected_points.backtest_workflow import historical_frame

frame = load_backtest_frame("path/to/frame.parquet")
save_backtest_frame(historical_frame(frame, 2025), "nfl", destination="path/to/new-frame.parquet")
```

This preserves structured quote columns and regenerates metadata. Prefer the normal command for automatic provenance
retention. Old bundles without checksums remain readable if row/column metadata matches; regeneration establishes
stronger integrity.

## Reference resolution

Baseline and candidate references accept `working-tree`, `deployed`, `version:N.N`, `sha:<sha>`, and `artifact:<path>`.
Deployed/version resolution uses the immutable live `model_releases.source_git_sha`, never a pick-update SHA. Both
frame preparation and fitting use that same resolved commit, even if a deployment happens during the comparison.
Routine retraining cannot move a version's baseline. Bootstrap 1.0's originally null SHA is initialized once by the
verified release workflow; a later canonical source SHA requires a new model version. References predating the
backtest protocol need saved run artifacts with matching input frames.

Temporary Git checkouts are removed after use. Dependency changes use fingerprinted environments under
`.backtests/expected_points/environments/`. Released/version/SHA cutoff caching is on by default. `deployed` and the
matching `version:N.N` share a readable `version-N.N` namespace; explicit SHAs have fingerprinted namespaces. Opted-in
working-tree caching uses one rolling namespace per league, and every entry still requires matching code,
configuration, historical inputs, dependencies, and cutoff metadata. `NO_CACHE=1` disables cutoff caches throughout.

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
