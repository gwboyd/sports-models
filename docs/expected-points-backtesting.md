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

Keep candidate code/settings, including notebook source cells, unchanged while a job runs. Make permanent feature and training changes in the source
used by the notebook/recipe so preparation and fitting reproduce them. After another model change, run the same
command again; automatic preparation generates new frames rather than silently reusing `latest.parquet`.

Training windows may differ **before the first evaluated season**. The command reports each side's training
seasons/counts, retains those rows for fitting, and requires the same game/outcome/market population from the first
evaluated season onward. It also checks that shared older games have identical source outcomes and markets.
There is no silent intersection. Original run artifacts keep full-history source fingerprints; paired comparison
views use a separately validated evaluation fingerprint. Missing or changed evaluation games still fail preflight.


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

Use these locations under Git-ignored `.backtests/expected_points/`:

| Location | Purpose |
|---|---|
| `comparisons/<league>/<timestamp>/` | One baseline-versus-candidate job, produced by the normal command. |
| `experiments/<dated-study>/` | Multi-variant research, study configuration, supporting analyses, and conclusions. Reference normal comparison jobs instead of copying them. |
| `reports/` | Consolidated model improvement, experiment, and optimization findings across jobs or studies. |
| `cache/<league>/<recipe>/` | Reusable cutoff results, validated before reuse. |
| `frames/` | Optional standalone exports; normal jobs save frames inside their job directory. |
| `runs/<league>/<unique-name>/` | Optional single-recipe CLI runs; normal comparison jobs already contain both sides. |
| `environments/` | Dependency environments for baselines that need them; created only when needed. |

A local `README.md` index can point to selected results and explain which older candidates were superseded. Keep
that index local and label the evidence's date/version; historical experiment conclusions must not be confused with
the latest selection. Workflow smoke checks should use temporary directories rather than ad hoc top-level folders.
When cleaning up, retain selected comparisons, referenced earlier runs, research evidence, and useful baseline caches.
Remove obsolete smoke checks and superseded preliminary jobs only after checking that retained reports/studies do
not reference them. Keep whole jobs and Parquet/JSON bundles together; never delete artifacts used by an active run.


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

## From an improvement idea to a release

Define the question, baseline, and fixed evaluation population before comparing variants. Implement in source with
focused chronology/correctness checks, then use the standard command for the final candidate. Research scripts can
screen explicit designs, but their narrower results must be labeled. Record failed and rejected variants as well as
winners, inspect uncertainty and per-season behavior, and distinguish correctness from measured accuracy.

After a selection, update `UNRELEASED.md` with public differences and the NFL/CFB How It Works Markdown with the
complete current methodology. Those public pages should cover inputs, training, predictions, confidence, update
behavior, and limitations, not over-focus on what changed in this release. Keep detailed evaluation evidence local.
If the user explicitly waives a new comparison after a prediction-affecting edit, state that the saved metrics predate
the edit in the existing evaluation report and index. Do not silently relabel old results or claim the fix improved accuracy.
Before the original commit/merge, run `make prepare-model-release` to choose versions and save the exact release notes,
empty drafts, and `model-versions.json`. Commit these with the model changes and How It Works updates, then merge once.
Deploy with `make sam-deploy` on reviewed, clean main; it reads the prepared versions and does not modify tracked files.
If prediction changes continue after preparation, copy the unpublished notes back to UNRELEASED.md and update the
consolidated draft, whole-model explanation, and relevant evidence. Any nonblank draft blocks deployment. Rerun prep
to amend the same unregistered version; changes after registration require a new version. Source/dependency signals
help catch missed updates and require explicit prediction-neutral classification when keeping a registered version;
they do not replace recording changes in the draft as they happen. Supabase determines when
a release actually deployed and became live; preparation and evaluation results alone do not deploy a model. The root
README documents retry and registration recovery.

## Manually written study and report files

Read `.backtests/expected_points/AGENTS.md` when present for local examples and folder-specific instructions.
That file and the local `README.md` index are Git-ignored; this tracked guide retains the conventions for fresh clones.
The normal command creates comparison artifacts automatically. It does not create a research plan, cross-variant
ledger, consolidated assessment, or update the local selection index; the person or agent doing the study owns those.

For a new study, create `experiments/<YYYYMMDD>-<study>/` without overwriting an existing study. A small convention is:

- `PLAN.md`: research question, variants, baseline reference, source provenance, training/evaluation windows,
  exclusions, fixed game population, profiles, and selection/confirmation criteria.
- `runs.csv`: one row per attempted variant, including settings, command, artifact path, coverage, status, and
  headline metrics. Include failures/rejected variants; unavailable metrics stay empty.
- `REPORT.md`: conclusions, uncertainty, regressions, selection effects, and links to completed evidence.

Normal comparisons remain in their generated job directories; record links rather than copying jobs. Advanced
research scripts require explicit inputs and a new `--output` directory within the study, which the script creates.
They produce requests, frames, and paired results, but the study author still writes the ledger and synthesis.
Use the research sections below to choose the appropriate runner; do not confuse frozen-input experiments with
source changes that regenerate all features. Reusable analysis logic belongs in tracked scripts/modules.

Write a new cross-study synthesis to `reports/<date>-<league-or-release>-<subject>.md`. Existing release-wide
report names can remain; date updates and label superseded selections explicitly. Include source/model references,
training and evaluation populations, principal metrics and intervals, season-level weaknesses, experiment-selection
limitations, and remaining validation. Derive tables from saved outputs, verify relative links, and update the local
`README.md` index with the selected report and final jobs. Candidate selection is not proof of deployment.
Do not edit generated reports/manifests to change the apparent outcome; put interpretation in these narrative files.

These saved reports are for model improvement, experiments, and optimization. Give code-review findings and routine
implementation/status summaries in chat unless the user requests a document. If a review reveals a limitation in
earlier performance evidence, annotate the existing evaluation report and index without creating a code-review report.

For manual frame output, use `save_backtest_frame(..., destination=..., provenance=...)` with a new descriptive path
and truthful construction details. Keep the Parquet and JSON together; see the inspection section below. Never edit
cutoff caches or dependency environments manually. Public release notes and How It Works pages stay separate from
these internal reports, including the runner's automatically generated `release_evaluation.md`.

## Efficient monitoring for agents

Start each job once and retain its process/session ID, job directory, and log path. Use completion notifications
when available; otherwise choose bounded waits based on the latest cutoff time and ETA instead of polling every few
seconds. In an active conversation, keep waits short enough to meet required user-update intervals. Read only new
session output or a short log tail; open full logs when diagnosing a specific failure. Work on independent tasks or
wait between checks without repeatedly reconsidering the same plan. Report milestones, failures, and decisions
briefly rather than narrating unchanged status.

A quiet notebook preparation or model fit is not evidence of a hang. Check process/session state and logs before
retrying, avoid overlapping retries, and reuse completed artifacts after a later failure. Do not run quick followed
by standard by habit: candidate fits are duplicated with default caching. Use quick only when its smaller coverage
answers an explicit screening question. Keep the default user workflow as the single standard command.

Release notes describe shipped model differences. Keep generated evaluation reports under Git-ignored
`.backtests/expected_points/reports/`, with detailed run artifacts in the existing comparisons/experiments
folders, not in tracked documentation. These reports are local-only and are not available in a fresh clone. Do not
force-add them; export outside Git when sharing is explicitly requested. On version changes, update the public
NFL/CFB How It Works Markdown to describe the released methodology and `UNRELEASED.md` to describe its differences;
neither should contain the internal evaluation report. The proposed 2.0 evidence is
in `.backtests/expected_points/reports/expected-points-2.0-evaluation.md` (local, Git-ignored), with links/paths to the detailed local reports.

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
Generated control/export cells preserve the source notebook schema, including legacy 4.4 notebooks, and are validated
before execution. The checked-in source notebook is not rewritten.

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

## Controlled opponent-adjustment research

The standard user/agent command remains `make backtest-expected-points LEAGUE=<nfl|cfb>`. For repeated research
on explicitly frozen inputs, `scripts/experiment_opponent_adjustment.py --help` exposes an advanced runner that
uses the same walk-forward fitter, full tuning grids, paired bootstrap metrics, and report writers. Supply a
completed deployed comparison with `--baseline`, scalar source snapshots with `--sources`, and a new `--output`.
It never fetches credentials or writes operational state.

Optional research controls are `--alpha` (ridge penalty), `--carryover` (prior-season weight), `--strength`
(correction fraction from zero to one), `--fcs-policy`, `--metrics` (registered CFB metrics), `--history-start`,
and `--exclude-seasons`. History controls affect efficiency features only; the saved baseline's score-training
and evaluation population stays fixed. To isolate older score-training games with frozen features, use the controlled training-history tool below.
To evaluate a source history-policy change with regenerated features, use the normal comparison command.
`--prepare-only` saves a source-fingerprinted feature frame without fitting. `--prepared-frame` reuses an explicit
frame. Alternatively, matching `--raw-frame` and `--adjusted-frame` endpoints can be linearly blended at
`--strength`: ratings, source observations, and smoothing must otherwise be identical. Unchanged features are
preserved exactly. Each job retains the request, source checksums, frame bundle, candidate run, and paired report.

The research runner defaults to quick cadence on explicitly selected 2023–2024 seasons; set `--seasons` for
another research window. Those are research defaults, not the standard comparison's date policy. It subsets a
completed baseline only at the requested cutoffs and verifies every evaluated game. Screen earlier seasons,
then confirm the chosen candidate using the ordinary standard command; do not select solely by lock win rate.
Report all variants and disclose that bootstrap intervals do not correct for selection across experiments.

CFB notebook `efficiency_metrics` controls the registered metric group, and `feature_history_start_year` controls
efficiency warmup independently of `training_start_year`. The defaults use overall explosiveness/PPA/success
rate, half-strength correction, pooled FCS opponents, 2018 score training, and 2017 warmup.
`excluded_history_seasons` defaults to `[2020]` and removes that season before loading, Elo, and efficiency calculations. NFL defaults to half-strength correction on its existing EPA/success inputs. Partial adjustment mappings
merge with league defaults. After changing any of these settings in source, the normal standard command prepares
the appropriate inputs automatically. Missing historical opponents must not silently discard loaded warmup stats.

The optional `smoothing_span` adjustment parameter overrides the base EWMA span (default: each metric's existing
10-game span); dynamic metrics still use the larger of that base span and the target week. Research CLI `--span`
sets the same lever. Changing it requires new candidate features and normal comparison evidence.

### Explicit training-history experiments

To isolate additional training rows from feature changes, the advanced read-only
`scripts/experiment_cfb_training_history.py` accepts `--reference-frame`, its completed
`--reference-run`, and an explicitly prepared `--history-frame` containing the same rows plus older games.
Every existing value must match exactly; only additional rows preceding the entire reference frame are allowed.
The normal command also supports different earlier training histories, while regenerating each recipe's features.
This controlled runner additionally freezes every existing feature value to isolate the training-sample effect.

Use `--starts 2018 2015 2013` to request training starts, `--exclude-seasons 2020` to omit COVID,
and `--output <new-directory>` for the experiment artifacts. The default is standard weekly coverage on
2023–2025; `--seasons` and `--profile` are optional overrides. Every requested start must actually contain
training games. Missing historical coverage fails instead of silently testing the same shorter history twice.
The historical frame must have a valid JSON sidecar recording its construction and source provenance.
Frame preparation must also exclude COVID from earlier feature/rating reconstruction if that is the design;
the runner's exclusion flag alone removes training rows, not already calculated historical features.

All variants preflight before fitting, use the shared chronological fitter and tuning grids, and retain original
full-history fingerprints in their saved run artifacts. The explicitly scoped paired reports compare a verified,
fixed evaluation population; `design.json` records both full-history identities and the evaluation identity.
Comparison views must not be used as source artifacts or fitting caches. Candidate fits are uncached by default.
Inspect feature coverage by season before fitting; a listed feed range does not guarantee usable sportsbook data.
The September 2026 coverage audit found only currently excluded `consensus`, `numberfire`, and `teamrankings`
provider labels in 2013–2017. The 2018 history has eligible sportsbook quotes. Do not silently treat earlier
non-sportsbook provider values as executable quotes; a proxy-training study needs an explicit source policy and
must retain the real sportsbook evaluation population. Provider labels alone do not establish historical provenance.
