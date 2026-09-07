# CFB Expected Points Model

The CFB expected-points notebook uses the shared expected-points training, tracking, grading, and Supabase
persistence workflow. CFB-specific acquisition and feature preparation remain in this directory. Production notebook
runs use strict validation, chronological training splits, and the direct CFBD client described in the repository
README and `AGENTS.md`.

The CFB recipe uses opponent-adjusted offense/defense explosiveness, overall PPA, and success rate.
Existing explosiveness names stay stable; new efficiency columns use the same descriptive EWMA naming convention.
The shared transform fits ridge offense/defense effects from strictly earlier games and applies half of the
estimated opponent correction before the established moving average. FCS teams share one pooled entity by default.
Ratings use the opening kickoff of each game week; exact per-kickoff snapshots remain an optional alternative.

Score training defaults to 2018, with 2017 efficiency warmup. `history.py` owns the explicit policy:
`training_start_year` overrides the first score-training season, `feature_history_start_year` overrides efficiency
warmup (default: one year before training), and `excluded_history_seasons` defaults to `[2020]`. Exclusions apply
before feed loading, Elo updates, and efficiency construction, so COVID games cannot enter any of those paths.
The separate `efficiency_schedule` supplies warmup opponents without making warmup games score-training examples.
Changing the history policy regenerates every feature under that policy; the earlier frozen-feature research
experiment is retained as separate evidence, not used as a production input artifact.

Notebook/Papermill `opponent_adjustment_config` accepts partial overrides of the league defaults, for example
`{"ridge_alpha": 10.0, "season_carryover": 0.25}`. Other controls are `adjustment_strength` (zero to one),
`fcs_policy` (`pooled` or `partial_pool`), `rating_snapshot`, and `excluded_seasons`. An explicit empty
`excluded_seasons` list allows every loaded season in the adjustment engine; it cannot restore seasons already
removed by `excluded_history_seasons`. Zero carryover with no current-season observations uses a
neutral correction. The dynamic moving-average policy is unchanged.

The registry also supports rushing/passing PPA and success-rate splits. Set notebook `efficiency_metrics` to an
explicit sequence of registered names to test another group; the notebook passes that same selection to the
adapter and `CFBExpectedPointsRecipe`. Requested missing features fail rather than silently disappearing.
The canonical observations are offensive per-play measures; their scheduled opponents supply the defensive side.
Registered names are `explosiveness`, `ppa`, `success_rate`, `rushing_plays_ppa`, `passing_plays_ppa`,
`rushing_plays_success_rate`, and `passing_plays_success_rate`.
Research results are recorded in `.backtests/expected_points/reports/expected-points-2.0-evaluation.md` (local, Git-ignored);
the backtesting guide describes experiment settings. `UNRELEASED.md` contains only public release notes.
Older score-training rows can be studied separately with the guide's
[explicit training-history experiment](../../../../../docs/expected-points-backtesting.md#explicit-training-history-experiments).
That runner freezes all existing inputs and evaluated games, excludes 2020 by default, and rejects requested
starts without usable rows. It is a controlled research tool; ordinary source history changes use the normal comparison command.

The notebook remains the interactive analysis surface. Its reproducible configuration lives in `recipe.py`, while
`schedule`, `market_lines`, intermediate feature frames, the complete `df`, `ep_config`, `results`, and `plays` remain
available for direct slicing. After training, `health_metrics` displays chronology-safe score, probability,
calibration, and weekly lock results. `inspect_game(game_id)` returns the market/model row, exact score and confidence
features, selected execution quote, prediction, and lock decision; `inspect_team_history` shows earlier team-game rows.

Interactive users may set `write_backtest_frame=True` and run through the dedicated frame-save cell without running
the normal model train. Training, game inspection, and comparison are separate stages. Set
`run_historical_backtest=True` only to launch `quick`, `standard`, or `full`; Papermill/API runs reject the expensive
option. The standard CLI command automatically prepares its own frames:
`make backtest-expected-points LEAGUE=cfb`. Persistent cutoff artifacts live under
ignored `.backtests/expected_points/` and are invalidated by recipe/configuration or relevant historical-input
changes. Working-tree fits are fresh by default; set notebook parameter `backtest_cache_working_tree=True` or Make
variable `CACHE_WORKING_TREE=1` only when resumable candidate caching is desired. CFBD has no point-in-time provider
snapshots, so reports label this as weekly historical approximation. See
the root [`backtesting guide`](../../../../../docs/expected-points-backtesting.md) for baseline references, expense
controls, artifacts, cache behavior, and metric interpretation.

`CFBExpectedPointsRecipe` owns final schedule/pregame-feature/market joins, eligibility filtering, kickoff and column
normalization, validation, feature selection, tuning grids, confidence inputs, betting transform, and lock thresholds.
The notebook keeps `joined_features` and `df` visible. Versions may use different feature columns automatically. Optional `BASELINE_FRAME`/`CANDIDATE_FRAME`
overrides reuse explicit saved inputs; both evaluation universes must still match.

## Picks Update Cadence

A small EventBridge-driven planner derives updates from the CFBD regular-season schedule rather than fixed CFB
weekdays. It keeps the current week active until its applicable games are final and the 5:00 AM Eastern rollover check
has passed on the following morning. A new season switches to active cadence inside the four-day horizon.

An eligible CFB week trains daily at 5:00 AM Eastern through its final game date. Every date containing games receives
up to three additional updates: one hour before the first early game, first afternoon game, and first night game.
Early means before 2:00 PM Eastern, afternoon is 2:00-6:59 PM, and night begins at 7:00 PM. Empty buckets are skipped,
and the same rule covers every weekday without maintaining separate cron rules. Multiple games within one window
share the update before its earliest kickoff.

The planner stores the week in `scheduled_model_updates` and creates or updates stable one-time EventBridge schedules.
Those schedules invoke the training Lambda directly through IAM. Kickoff changes overwrite the future trigger instead
of adding one, and the training Lambda atomically claims its deterministic run identity before Papermill starts. Its
database transaction links the plan's generic `(model_key, update_id)` fields to the persisted CFB update row. It never
uses API Gateway or an API key. With no published future game, weekly schedule-feed checks create no training run.
Once next-season games are published outside the four-day horizon, one weekly offseason training run is created;
inside four days, active cadence replaces it. Final-season picks are graded by that next successful notebook run, not
by the coordinator.

## On-demand refresh

`POST /cfb-update-picks` on the deployed API takes no body (or `{}`) and requires admin `Authorization` and
`client-name` headers. It selects the current eligible slate and schedules the existing trainer 60–120 seconds later.
A confirmed submission returns `202` with the selected season/week, run key, and status URL; no eligible slate returns
`200` / `not_scheduled`. Explicit-week bodies and the reserved client name `notebook` return `422`.

Supply an `Idempotency-Key` for safe retries. Reusing it retrieves the same job; a new key requests another run.
Poll admin-only `GET /model-update-jobs/{run_key}` for state, errors, and the completed update's version/SHA and counts.
Manual jobs retain the requesting client and use the trainer's version at execution time. Older protected picks keep
their original versions.

See [On-demand updates](../../../../../README.md#on-demand-updates) for request examples, retry/status semantics,
and the required database additions and deployment checks under [Scheduled Pick Updates](../../../../../README.md#scheduled-pick-updates).

## Model Release Queue

Keep the public [How It Works explanation](../../../../../frontend/content/cfb-how-it-works.md) current in
the same change as each model version's release notes. It describes the whole model (inputs, training, predictions, confidence, updates, and limitations), with ridge
adjustment as one part; `UNRELEASED.md` describes only user-visible differences. Review the other league's page
when shared behavior changes and coordinate frontend publication with the backend release. Keep large reports in
Git-ignored `.backtests/expected_points/reports/`, with detailed comparisons/experiments in their existing folders.
Those local reports are not shipped or pushed; do not force-add them.

Author CFB release notes in `UNRELEASED.md`; an empty draft means no unprepared changes, not necessarily no prepared
release. Do not add frontend, API-output, infrastructure, or database-only changes to model notes. Use a `#` title,
a plain-language `## Public Summary`, and `## Changes` with 2–3 concise bullets giving meaningful technical detail
(estimator, chronology, shrinkage, thresholds, or bet accounting). Optionally add one baseline-comparison bullet with
verified results for the actual recipe, the period/population, assumptions, and material uncertainty or selection
limitations. Keep complete evidence in Git-ignored `.backtests/expected_points/reports/`; omit source paths, commands,
experiment ledgers, and separate evaluation/internal headings. See the root README for the complete writing guidance.

Before committing/merging a model release, run `make prepare-model-release` on the feature branch. It reads
registered versions, prompts for major/minor for each populated NFL/CFB draft, and confirms before copying the exact
notes to `releases/vN.N.md`, clearing the drafts, and updating root `model-versions.json`. Commit these files and the
updated whole-model How It Works pages with the implementation, then merge once. Preparation writes no Supabase rows
and does not deploy. For any revision to unpublished prepared notes, including wording-only edits, copy the complete
notes back into UNRELEASED.md and edit there; do not hand-edit the archive or manifest. Review How It Works and update
it if behavior changes, then rerun prep. After confirmation it amends that same
unregistered version and clears the draft. Any nonblank draft blocks deployment. After registration, new prediction
changes require a new major/minor version. Do not manually clear drafts to bypass these checks.

Run `make sam-deploy` on clean `main` to deploy the committed versions. It rejects unprepared drafts, stale versions,
and changed registered notes, prints both versions, and requires confirmation. Both commands show changed source
and dependency inputs since the immutable registered SHA. Keeping the version despite those signals requires an
explicit prediction-neutral classification; they never choose a bump automatically. After SAM succeeds it verifies the
API, training, and coordinator Lambdas, registers/verifies releases in Supabase, and clears its ignored recovery plan.
It does not modify tracked files, so there is no second documentation merge. Existing canonical SHAs remain fixed;
a null bootstrap SHA is initialized only once. Supabase records actual deployment metadata and marks a release live
at its first successful AWS pick update (`first_pick_at`). The manifest and notes alone only indicate preparation.

For a failed build/deploy, rerun `make sam-deploy` at the same clean commit to resume its plan. After AWS succeeds,
Deployment rechecks the registry after confirmation and build to reject competing releases. Recovery validates both
league snapshots and their note hashes before verifying AWS or registering releases.
`make sam-register-releases` can recover registration without rebuilding or changing tracked files. See the root
README for recovery and preparation details. Prediction-neutral changes keep the manifest versions and empty drafts.


Only the deployed AWS training Lambda writes automatically. Interactive runs default to `client_name="notebook"`
and `allow_non_aws_write=False`, so they remain read-only. To perform an intentional notebook write, set
`allow_non_aws_write=True`; the notebook resolves the latest registered CFB version and requires the exact
`WRITE CFB <VERSION>` confirmation before using the shared atomic writer. Local API and SAM-local update requests
return `403` and cannot create production schedules. Explicit season/week and non-AWS write options remain available
to the internal notebook runner, not the HTTP endpoint.

### Evaluating local changes against deployed

**For any prediction-affecting change, just run `make backtest-expected-points LEAGUE=cfb` from the repository
root.** Adding, removing, renaming, or recalculating features uses the same command as changing training, estimators,
confidence inputs, or lock rules. It automatically prepares baseline and candidate features using each version's
code, validates their common historical evaluation universe, and runs the standard three-season weekly comparison.
It saves reports, prediction/lock changes, input bundles, notebooks, and logs in a new timestamped job directory.

No notebook run, `prepare-frame`, preflight, or quick run is required beforehand. `PROFILE=quick` is an optional
smaller run and repeats overlapping candidate fits if followed by standard with default caching. Optional controls
for dates/seasons, caching, saved frames, alternate baselines, and recovery are described in the
[backtesting guide](../../../../../docs/expected-points-backtesting.md#evaluate-working-tree-changes-against-deployed).
Keep code/settings fixed during evaluation and run the same command again after another model change. Opponent-adjusted
dynamic smoothing uses the target game's week for its span; older frames need regeneration after that correction.

The optional `smoothing_span` adjustment parameter overrides the base EWMA span (default: each metric's existing
10-game span); dynamic metrics still use the larger of that base span and the target week. Research CLI `--span`
sets the same lever. Changing it requires new candidate features and normal comparison evidence.

## Version 2.1 betting probabilities

Spread probabilities use `q=.5+.1*F(e)`, where `e` is the absolute execution-line edge and `F` is the absolute margin
error CDF on the chronological score holdout. Symmetric errors and the 80% discount toward even chance are modeling
assumptions. Totals use a shrunken historical rate with Locks disabled. Locks require q>=.525, positive expected units
at assumed -110, and two corroborating sportsbook quotes. The normal combined limit is three spreads; q>=.55 permits
extras without a hard five-bet ceiling. Started Locks retain their version/status and count toward weekly capacity.

## Lightweight Lock replay

Use `make replay-expected-points-locks LEAGUE=cfb` to anchor to the deployed release's latest matching local
standard score run; `LOCK_SOURCE=version:2.0` or `artifact:<run>` pins another source. No expected-score refit is
needed. `LOCK_SCREEN_CADENCE=1 LOCK_MIN_PROBABILITY=.525` compares combined weekly volume/profit policies and
labels all evaluated seasons as exposed research. New standard runs retain checksummed per-cutoff
`lock_training.parquet` data; CLI `--training-history score-holdout --variants symmetric_residual_0.2` replays
those exact calibration inputs. Older caches support weekly-history research, not identical production calibration
history. The [backtesting guide](../../../../../docs/expected-points-backtesting.md#lightweight-lock-method-replay)
documents controls, provenance, uncertainty and mandatory production-path comparisons.
