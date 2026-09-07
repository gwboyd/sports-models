# Project Overview
This repository hosts sports prediction systems with a FastAPI backend deployed to AWS Lambda via
Mangum/SAM, notebook-driven model workflows for NFL, CFB, and NBA, and a frontend for model consumption.
Operational persistence now lives in Supabase Postgres under the `sports_models` schema. Expected-points picks and
graded results retain the recipe version that generated the pick, including when an older locked pick is graded after a
newer deployment.

## Repository Structure
- `src/model_patterns/`: Reusable modeling patterns; `expected_points/` is currently shared by NFL and CFB.
- `src/model_patterns/expected_points/backtesting.py`: Walk-forward cutoff orchestration, cache identity, and fitting.
- `src/model_patterns/expected_points/backtest_metrics.py`: Historical summaries, uncertainty, and comparisons.
- `src/model_patterns/expected_points/backtest_artifacts.py`: Parquet/JSON/Markdown persistence and frame loading.
- `src/sports/`: Sport- and model-specific implementations (NFL/NBA/CFB notebooks, handlers, utilities).
- `src/sports/football/schedule_coordinator.py`: Dynamic calendar policy and lightweight Lambda entrypoint.
- `src/sports/football/scheduled_updates.py`: Idempotent direct training-Lambda dispatcher.
- `src/sports/football/manual_updates.py`: Current-slate HTTP submission and read-only job-status presentation.
- `src/utils/db/`: Centralized Postgres data access for operational tables/views.
- `src/utils/`: Shared infrastructure utilities and Pydantic models.
- `frontend/`: Frontend app for serving model views.
- `tests/`: Python test modules for API/Lambda behavior.
- `events/`: Sample AWS SAM event payloads for local invocation.
- `scripts/backtest_expected_points.py`: Read-only NFL/CFB weekly walk-forward runner and baseline comparator.
- `scripts/deploy_models.py`: Local release preparation, clean-main deployment, and registration recovery.
- `model-versions.json`: Committed intended NFL/CFB versions; actual deployment/live status remains in Supabase.
- `python/`: Vendored Python packages for Lambda packaging/layer-style usage.
- `main.py`: FastAPI app entrypoint, auth/permissions, router mounting, and Mangum handler.
- `template.yaml`: AWS SAM template for the Lambda function and API event wiring.
- `Dockerfile`: Lambda-compatible container image build definition.
- `requirements.txt`: Python dependency list for backend/model runtime.
- `requirements-dev.txt`: Runtime dependencies plus pinned local test tooling.
- `db/sql/001_create_sports_models_schema.sql`: Supabase setup SQL for the `sports_models` schema.

## Build & Development Commands
```sh
brew install uv
uv venv
source .venv/bin/activate
brew install libomp
brew install python-certifi
uv pip install -r requirements-dev.txt
uv pip check
```

```sh
uvicorn main:app --host 0.0.0.0 --port 3000 --reload --log-level warning
```

```sh
sam local invoke "ApiLambdaFunction"
sam local start-api
sam local invoke ApiLambdaFunction -e events/get-health-event.json
```

```sh
pytest
```

```sh
cd frontend
npm install
npm run dev
npm run build
npm run typecheck
npm run codegen
```

## Code Style & Conventions
- Python: keep type hints on public surfaces, explicit logging, and small focused helpers.
- SQL access belongs in `src/utils/db/`, not inline in handlers.
- Notebook runtime cells should stay separated from notebook-only exploratory cells using explicit guards.
- TypeScript frontend uses strict mode.
- Python modules/files use `snake_case`.
- React component files use `PascalCase`.

## Architecture Notes
```mermaid
flowchart LR
  U[Client Browser] --> FE[Frontend]
  FE --> API[FastAPI on AWS Lambda]
  CLOCK[Recurring EventBridge Schedule] --> COORD[Small Schedule Planner Lambda]
  COORD --> PLANS[(Supabase Update Plans)]
  COORD --> SCH[One-time EventBridge Schedules]
  SCH --> JOB[Direct Training Dispatcher]
  API --> NFL[NFL Router]
  API --> CFB[CFB Router]
  API --> NBA[NBA Router]
  NFL --> SCH
  CFB --> SCH
  JOB --> EP[Shared Expected Points Runtime]
  EP --> NB[Papermill Notebook Execution]
  NB --> DB[(Supabase Postgres)]
  NFL --> DB
  CFB --> DB
  NBA --> DB
  NB --> TMP[/tmp outputs]
  API --> RESP[JSON Responses]
```

`main.py` creates the FastAPI app, validates API keys, and mounts sport/model routers. Operational reads and
writes go through `src/utils/db/sports_models_db.py`. NFL and CFB update endpoints on the API Lambda create one-time
Scheduler jobs; the training Lambda executes their notebooks via the shared runtime in
`src/model_patterns/expected_points/runtime.py`. Shared tracking helpers validate picks, preserve started games, calculate changes, and grade completed picks. `write_expected_points_run` persists the update record,
current picks, and newly graded results in one transaction so a failed run cannot partially commit. For a scheduled
run, that same transaction links `scheduled_model_updates.update_id` to the model-specific update-history row and
marks the plan completed. Initial Postgres connection failures use bounded retries in `src/utils/postgres.py`.

EventBridge invokes a 512 MB coordinator Lambda at 4:45 AM and noon Eastern. Its separate image command runs
`src/sports/football/schedule_coordinator.py` from the same repository image without importing the FastAPI handler.
The coordinator checks Supabase state before loading nflverse/CFBD, backs off to weekly checks when no game is within
the active horizon, derives exact run windows from schedule dates, and reconciles named one-time EventBridge Scheduler
resources. Those one-time schedules invoke the training Lambda through a dedicated IAM role; the coordinator never
invokes training itself. `main.handler` sends `expected_points_update` events to `scheduled_updates.py` and HTTP events
through Mangum. Scheduled exceptions must propagate so asynchronous retries and SQS failure destinations can observe
them. The training Lambda has reserved concurrency of one, and the API, coordinator, and trainer share a verified Git SHA.

NFL expected-points acquisition is isolated in `src/sports/football/nfl/expected_points/data_loader.py`: `nflreadpy`
loads source data as Polars, selects model fields, and converts to pandas at the notebook boundary with caching off.
CFB expected-points uses `src/sports/football/cfb/expected_points/cfbd_client.py` for direct authenticated REST
requests; do not add a conflicting Python SDK dependency.
NFL quarterback features must use only starts strictly earlier than the target kickoff. Debut/unknown history stays
missing; never calculate a fallback using all future players or seasons. Native score-model missing handling and
confidence imputation fitted within chronological training data handle those inputs. Explicit NFL/CFB efficiency
selections must fail when any selected column is missing; never fall back to substring-based column discovery.
Opponent-adjusted football efficiency features are implemented in
`src/sports/football/transforms/opponent_adjustment.py`. It fits pre-kickoff ridge offense/defense effects, then
recomputes each team's historical performance against the opponent it faced before applying the established smoother.
Keep downstream feature names stable. Ridge strength, season carryover, CFB's FCS pooling policy, and rating snapshot
cadence are versioned recipe/backtest levers; any change requires regenerated candidate frames and normal baseline
comparison evidence. The correction strength is independently configurable from zero to one; excluded seasons
are removed from both rating and smoothed efficiency history, without silently changing score-training rows. CFB
efficiency metrics are explicitly registered/selected in its features and recipe modules; selected missing columns
must fail. Research screens use fixed game populations and the existing walk-forward fitter/paired reports, with
later-season confirmation and disclosure of experiment selection. The current candidates use half-strength
correction. CFB selects explosiveness, overall PPA, and success rate with pooled FCS effects; its separate efficiency
schedule retains 2017 warmup opponents while score training starts in 2018. `CFBHistoryConfig` in `history.py`
owns the training start and historical exclusions. Notebook overrides are `training_start_year`,
`feature_history_start_year`, and `excluded_history_seasons`; 2020 is excluded before feed loading, Elo updates,
and efficiency calculations by default. Notebook metric selections must match the recipe; partial parameter overrides preserve
league defaults.

Current expected-points schema model, mirrored for the `nfl` and `cfb` prefixes:
- `<league>_expected_points_picks`: latest pick per `(year_week, game_id)`
- `<league>_expected_points_latest_picks`: view over the most recent `year_week`
- `<league>_expected_points_pick_updates`: update-run history and JSON snapshots
- `<league>_expected_points_latest_updates`: latest update per `year_week`
- `<league>_expected_points_results`: graded outcomes
- `model_releases`: immutable NFL/CFB expected-points recipe releases and public/internal release notes
- `schedule_coordinator_state`: per-league next-check timing for automatic active/offseason backoff
- `scheduled_model_updates`: backend-readable plan and execution history for named one-time AWS schedules; its
  `(model_key, update_id)` pair conditionally identifies the resulting model-specific update-history row

Expected-points recipe releases use independent `MAJOR.MINOR` versions for NFL and CFB. A major release changes
the model's methodology or expected behavior; a minor release is a smaller prediction-affecting change. Routine
retraining on newly available data is a run under the same version. Historical rows are attributed to the bootstrap
baseline `1.0`. Its archived release files describe the production recipes present when tracking began and explicitly
note that exact pre-versioning code revisions cannot be reconstructed. Picks, results, and update runs carry the model
version that produced them, so performance can be grouped by exact version or by major version. Locked picks preserve
their original version when later runs change the recipe.

CFB pick and result rows additionally retain nullable `home_conference` and `away_conference` values from the
CFBD schedule. The CFB frontend renders each game once and uses those fields for a conference filter; a
cross-conference game matches either team's conference.

The frontend is mobile-first and game-centered for NFL and CFB. Each current-slate page orders favorites, separate
spread/total lock cards, and the complete game list. Favorites are device-local under
`sports-models:favorites:v1`. Shared results pages calculate season and weekly summaries from existing graded game
responses. Both leagues have public methodology routes; NFL also has a model-insight route. Approved team logos are local
assets referenced by generated frontend manifests; missing assets use monogram fallbacks. Refresh CFB metadata and
assets with `make sync-cfb-teams YEAR=<season>` (using the root `CFBD_API_KEY`), NFL assets with
`make sync-nfl-teams`, or both with `make sync-football-teams YEAR=<season>`. The sync never writes API credentials to
frontend output; generated manifests live under `frontend/app/generated/` and cached assets under
`frontend/public/teams/<league>/`.
Frontend additions should preserve the compact design system: neutral surfaces, eight-pixel card radii, restrained
shadows, custom electric ink blue (`#0B5FCC`) interaction/lock accents, tabular numbers, and 44px minimum interactive targets.
Lock emphasis is a uniform medium-blue outline without an accent side border. Favorite game cards show that outline
and independent spread/total lock tags whenever either market qualifies.
Keep the football hero limited to the game-search field; favorite management belongs in the Favorites section. The
team manager uses a separately scrolling results region that resets to the top on each query so single matches remain
visible above the mobile keyboard. Shared mobile overlays must remain anchored to `window.visualViewport` and lock
document scrolling for the full open lifetime. Mobile lock rails must align their first card to the main content inset.
Football kickoff labels must render in the browser device timezone. NFL and CFB both expose `date_time` as
`America/New_York` wall time in `YYYY-MM-DD-HH:MM` format; convert raw CFBD UTC timestamps before persistence.
Model-update timestamps also render in the device timezone and include the applicable daylight/standard abbreviation.
Keep the NBA route functional while its global tab is temporarily hidden; NFL and CFB are the visible league tabs.
Results summary cards lead with spread/total locks and omit a standalone predicted-games tile. In favorites and mobile
game cards, emphasize the actionable spread/total picks and outline each locked market individually in the lock blue.
When a graded result matches a current pick by `year_week` and `game_id`, its individual market outline must override
lock styling with green for a win or red for a loss across favorite, lock, mobile, and desktop presentations. Standard
picks remain unfilled, while locks retain a label and add a light outcome-colored fill; use the same distinction with
a neutral treatment for pushes and state outcomes in the expanded model details rather than adding result badges.
For a graded current-slate game, replace kickoff information with a compact `Final` label and away/home final score in
favorite, lock, mobile, and desktop presentations.
Treat current-page result enrichment as supplemental so an upstream results failure does not prevent current picks
from rendering; dedicated result routes should continue to surface non-404 failures.
Finalized spread markets must state the result from the picked team's perspective, such as `Final: SAC lost by 11`;
finalized totals use only `Final: <combined score>`. Remove pregame-only detail from final favorite and lock surfaces
while retaining model projection context in expandable or hover details.
Public methodology lives in `frontend/content/nfl-how-it-works.md` and `frontend/content/cfb-how-it-works.md`,
served at each league's `/models/<league>/how-it-works` route and linked from its navigation. Both use the shared
`ModelMethodology` renderer, which must support headings through `h3`, lists, links, inline code, and responsive
images. Explain the model in plain language with useful mathematical detail, including ridge opponent adjustment,
season-history policy, and league-specific limitations. Keep deployment/database operations in developer READMEs.
Write How It Works as an explanation of the complete current model, not a release announcement. Cover inputs,
training and validation, matchup predictions, confidence/Locks, history and opponent handling, update behavior, and
limitations in proportion to their role. Keep the version as context; put the list of changes in `UNRELEASED.md`.
On every model version change, review and update the applicable public methodology in the same change as the
release draft. Keep its stated methodology version, features, training history, opponent policy, confidence/lock
rules, and refresh timing consistent with the recipe being released. Remove shipped features from future-ideas
lists. Review both leagues when shared behavior changes, and coordinate frontend publication with the backend
release so the public page does not claim an unreleased recipe is already live. Preserve useful existing explanations
and diagrams while correcting stale claims.

Only the deployed AWS training Lambda writes expected-points records automatically; the coordinator writes scheduling
state and plans but never picks, update history, or grading rows. Runtime origin is determined from Lambda runtime
markers while explicitly excluding `AWS_SAM_LOCAL`; `client_name` is audit metadata, not authority.
Local API and SAM-local update requests return `403` and cannot schedule production work. HTTP updates no longer
accept season/week or `allow_non_aws_write`; explicit parameters remain internal notebook-runner inputs. Notebooks default to
`client_name="notebook"` and `allow_non_aws_write=False`; enabling the flag requires the exact interactive
`WRITE <LEAGUE> <VERSION>` confirmation. The shared database writer must retain its defensive non-AWS authorization
check. CFB eligibility is explicit:
at least one participant must be FBS and at least one real sportsbook must supply each of the current spread and total;
moneylines, opening lines, and multiple providers are optional. CFB score features use median sportsbook consensus
lines. Betting direction is chosen against consensus, then the execution line is shopped only within the largest
provider cluster spanning at most two points. Cluster ties prefer proximity to consensus and then provider
reliability. Unsupported exact quote ties use Bovada, William Hill, ESPN Bet, DraftKings, Caesars, then remaining
providers alphabetically. A market without two corroborating providers remains visible and trainable but cannot
become a lock. The execution line drives model edge, confidence, persistence, display, and grading. Synthetic CFBD
providers are not executable quotes, and CFBD provider ordering must never affect selection.
Neither football model may predict a game at or after kickoff, including a game with no pre-existing pick.

Expected-points training uses an outer chronological score holdout, a single inner chronological GridSearch split,
and one final refit on all completed games with the selected score parameters. Version 2.1 Lock heads use the outer
out-of-time score-error distribution: NFL spreads/totals and CFB spreads use `q=.5+.1*F(abs(edge))`, where F is the
empirical absolute-error CDF; CFB totals use a shrunken base rate with Locks disabled. Legacy confidence classifiers
retain their chronological tuning path. Do not replace chronology with random splitting or rerun score GridSearch.
Locks require q>=.525 and positive EV at assumed -110. The normal combined weekly limit is three: NFL up to two
spreads and one total, CFB spreads only. q>=.55 can exceed positive market/combined caps; a zero market cap remains
hard-disabled. The aim is 2–3 combined Locks most weeks, not a quota and not 2–3 per market. Preserve started Locks
and count them toward capacity even when their games are absent from the new prediction slate. Shared replay and
production use the same head factory and selector. New historical runs save checksummed per-cutoff score-holdout
frames for exact-head replay; older caches remain usable with explicitly disclosed weekly-history approximation.
`lock-replay --screen-cadence` is retrospective policy research, not untouched confirmation or automatic promotion.
Legacy CFB confidence experiments score candidate parameters with log loss and always include the execution-line spread/total
edge. Opening-line, movement, market-depth, and no-vig moneyline feature groups remain out of production unless a
genuine out-of-time comparison improves pooled Brier score by at least one percent, does not worsen log loss, and
keeps every evaluated season within two percent of its baseline Brier score. Read-only CFB notebook runs execute and
report this secondary chronological classifier evaluation; production/API notebook runs skip it. No optional market
feature group is currently promoted.

Expected-points evaluation has two shared tiers. Each normal train retains the outer chronological score holdout and
adds an untouched confidence/lock health tail; top-N health locks reset inside each historical season/week. The
read-only walk-forward runner consumes a notebook-saved model frame and retrains an unfitted versioned recipe before
historical weekly slates. `quick`, `standard`, and `full` profiles change cutoff coverage, not per-cutoff tuning
fidelity. Local frames, appendable cutoff caches, prediction Parquet files, and reports live under ignored
`.backtests/expected_points/`; a corrected historical input invalidates that cutoff and every later expanding-history
fit. Recipe keys hash the Git SHA, relevant working-tree content, configuration, protocol, and dependency files.
Shared expected-points, selected-league, and football-transform Python files are discovered recursively for this
identity; do not replace discovery with a filename allowlist that can miss a new dirty or untracked helper.
Working-tree cutoffs occupy one rolling namespace per league: changed fingerprints retrain and atomically replace
the applicable entries instead of accumulating one namespace per local edit. `deployed` and `version:N.N` share a
readable `version-N.N` namespace whose entries still require full metadata matches; explicit Git references retain
separate fingerprinted namespaces.
Working-tree cutoff caching is disabled by default and must neither read nor write cache entries unless the CLI
`--cache-working-tree`, Make `CACHE_WORKING_TREE=1`, notebook `backtest_cache_working_tree=True`, or equivalent
`BacktestSpec` opt-in is explicit. Released/version/SHA baseline caching remains enabled by default; `NO_CACHE=1`
disables caching for the entire comparison.
Baselines resolve from deployed/version Git SHAs or explicit artifacts, execute in isolated worktrees, use cached
fingerprinted environments when dependencies differ, and never use estimator pickles. League recipes own final
model-frame assembly/validation; version-specific baseline and candidate frames may differ in feature columns but
must hash to the same game/outcome/market evaluation universe.
Deployed/version resolution must use the immutable `source_git_sha` stored on the live `model_releases` row. Routine
retraining and later prediction-neutral commits under the same version must not move that baseline; a new canonical
recipe SHA requires a new model version. Because bootstrap 1.0 predates SHA tracking, the first verified
protocol-capable deployment initializes its null registry SHA exactly once during release registration. Pick-update
SHAs remain per-run audit metadata and must never participate in baseline resolution.
Comparisons require identical profile, season, and game-key universes and use fixed-seed season-week block-bootstrap
intervals for score, market, confidence, calibration, coverage, and lock-frequency output. Cached and report Parquet
round trips must preserve structured trace columns, and notebook comparison objects expose lock changes directly.
Zero-decision win rates are undefined/null rather than zero percent. Single-run reports include per-season records,
absolute block-bootstrap intervals, confidence AUC/calibration diagnostics, model-versus-market MAE benchmarks, and
per-cutoff elapsed/cache timing. The CLI must log cutoff progress and ETA so long runs are observable.

NFL and CFB notebooks remain first-class inspection surfaces. They expose schedules, intermediate feature frames,
the complete model `df`, configuration, health results, current predictions, and picks. Notebook helpers trace exact
score/confidence inputs and earlier team rows for a game. Full comparisons are guarded by
`run_historical_backtest=False` and must never run in Papermill/API mode. Backtests use currently available historical
feed values and must be described as weekly approximations, not exact intraday snapshot replays. They never write
operational picks, update history, results, or releases to Supabase. `write_backtest_frame=True` belongs to a separate
interactive stage before training so a user can assemble and save `.backtests/expected_points/frames/<league>/latest.parquet`
without fitting the current model. Training, inspection/experiments, and historical comparison remain separate cells;
experiment frames/configuration must be copies rather than aliases. The canonical operating guide is
[`docs/expected-points-backtesting.md`](docs/expected-points-backtesting.md#evaluate-working-tree-changes-against-deployed).
For **any prediction-affecting NFL/CFB change**, the standard workflow is simply
`make backtest-expected-points LEAGUE=nfl` (or `LEAGUE=cfb`). This includes adding/removing/renaming features,
changing calculations, training windows, estimators, tuning, confidence, and lock rules. The command automatically
resolves/pins the deployed baseline, prepares each version's features in its own source context, preflights input
compatibility, runs the standard three-season weekly comparison, and saves reports/logs in a new timestamped job.
Do not require users or agents to run notebooks, prepare-frame, preflight, or quick first. `PROFILE=quick` is an
optional smaller run, not a prerequisite; with working-tree caching off it duplicates candidate fits if followed by
standard. Baseline/candidate feature schemas may differ; the historical game/outcome/market evaluation universe must
match. A change to that population requires an explicit evaluation design rather than a silent intersection.
Automatic preparation defaults to a conservative closed-season bound (previous year from March onward, two years
prior in January/February), or the last explicitly requested `SEASONS`; `THROUGH_SEASON` overrides it. Earlier training
history is retained. Explicit frame paths bypass preparation for that side; never silently use `latest.parquet`.
CLI `prepare-frame` is an optional inspection tool that stops before training. `PREFLIGHT_ONLY=1` prepares/checks
inputs without fitting. The guide documents date/context overrides, frame reuse, caches, artifacts, and recovery.
Generated preparation cells must remain valid for the source notebook's schema, including older release notebooks
without cell-ID support; validate the generated notebook before execution without changing its executable source.
Keep candidate code/settings fixed throughout the job, including notebook cells; preparation and fit-time
provenance checks reject stale code. Retain Parquet/JSON
pairs and regenerate metadata with `save_backtest_frame(..., destination=...)` after filtering. Finished baseline and
candidate runs survive later failures and can be reused with `artifact:<path>`; comparison artifacts automatically
locate their saved sibling frame bundles. Inspect actual process/session state and logs before retrying quiet runs,
and avoid overlapping retries. Update the guide whenever commands, defaults, cache behavior, or metrics change.
The improvement workflow is: define the question and evidence population; implement the candidate with focused
correctness/leakage checks; use the standard comparison unless an explicit research design needs a different profile;
record every attempted variant and uncertainty in local studies/reports; select based on the complete evidence;
update public release notes and the whole-model explanation; run `make prepare-model-release` before the original
commit/merge; complete the required review and clean-main deployment. The detailed optimization and release-preparation
workflow is in [`docs/expected-points-backtesting.md`](docs/expected-points-backtesting.md#from-an-improvement-idea-to-a-release).
If the user explicitly waives a fresh comparison after a prediction change, record that gap in the existing evaluation report
and index. Do not present earlier results as measurements of the changed recipe or assume a correctness fix improves
accuracy. Later comparisons must rebuild from the current source and preserve the deployed baseline unchanged.
For token-efficient simulation/backtest runs, launch each job once with a retained process/session ID and log path.
Use the runner's cutoff progress and ETA to time checks; use completion notifications or bounded waits instead of
rapid polling. Read only new output or a short log tail, and inspect full logs only to diagnose a failure. Between
checks, do useful independent work or wait without repeated analysis. Keep required user updates brief and focused
on milestones, failures, or decisions. Do not start overlapping retries or quick runs merely to fill waiting time;
standard remains the default. Preserve completed run artifacts for recovery rather than repeating successful fits.

League-level validation belongs in `src/sports/football/<league>/data_validation.py`; generic dataframe contracts
belong in `src/sports/data_validation.py`. Validate only feeds a model actually consumes. Production notebook runs use
`strict_data_validation=True`; warning mode is for feed investigation and must never weaken chronological/leakage
checks. Keep stable loading, validation, feature construction, and training mechanics in Python modules while using
notebooks for orchestration, exploration, charts, and persistence previews.
League-neutral football transforms belong in `src/sports/football/transforms/`; source-specific conversion and feed
shaping belong beside the NFL or CFB workflow. Shared NFL/CFB HTTP schemas belong in
`src/sports/football/expected_points_schemas.py`.

Prediction-affecting expected-points work must be recorded in the applicable
`src/sports/football/<league>/expected_points/UNRELEASED.md`. The file is intentionally empty when there are no
unprepared model changes. Before commit/merge, prepare a non-empty draft as a major or minor release; it cannot ship
under the old version. After preparation, pending release notes live in `releases/vN.N.md` and the intended version
lives in root `model-versions.json`, even though `UNRELEASED.md` is empty.

Keep the three documentation surfaces distinct:

- `UNRELEASED.md` explains what changes for users. Use a title, `## Public Summary`, and `## Changes` in plain
  language. Omit code references, experiment controls, evaluation results, and internal implementation details.
  The parser supports optional evaluation/internal sections for compatibility; do not add them to new drafts.
- Public How It Works pages explain how the applicable version operates, including useful mathematical detail and
  limitations. Update them alongside release notes as described above; they are not experiment ledgers.
- Consolidated model improvement, experiment, and optimization reports belong in Git-ignored
  `.backtests/expected_points/reports/`; detailed run artifacts remain in the comparisons/experiments directories.
  They retain metrics, uncertainty, logs, and evidence relevant to those studies. Give code-review findings and routine
  implementation/status summaries in chat unless the user requests a saved document. Keep limitations affecting prior
  performance evidence in the existing evaluation report/index; do not create a separate code-review report.
  Never force-add reports or copy them into tracked documentation. They are local-only and do not accompany
  a clone; explicitly export outside Git when sharing is requested. The 2.0 report is
  `.backtests/expected_points/reports/expected-points-2.0-evaluation.md`.

Before reading or writing local evaluation artifacts, read `.backtests/expected_points/AGENTS.md` when present.
It describes directory ownership, result inspection, and manual study/report writing. It is Git-ignored like the
artifacts; the tracked `docs/expected-points-backtesting.md` remains the durable guide for fresh clones.

Use `comparisons/<league>/<timestamp>/` for individual evaluations and `experiments/<dated-study>/` for research
across variants; studies should reference normal comparison jobs instead of copying them. Keep a local
`.backtests/expected_points/README.md` index for selected evidence. Put workflow smoke checks in temporary directories
rather than new ad hoc top-level folders. Cleanup must preserve selected runs, referenced research, baseline caches,
and complete frame/metadata bundles; check references and active runs before removing superseded artifacts.

Before merging a model release, `make prepare-model-release` saves the exact draft under `releases/vN.N.md`, clears
the working file, and updates `model-versions.json`. Commit these with the model changes and applicable whole-model
How It Works pages. Documentation, frontend, API-output, database, and infrastructure-only changes do not belong in
the draft and keep the prepared versions unchanged.

`.dockerignore` must exclude `.backtests/` as well as `.env` and other local artifacts. Git-ignore rules alone do
not keep reports, snapshots, caches, or environment directories out of Docker's `COPY . .` build context.

The release workflow is **prepare on the feature branch, commit/merge once, then deploy clean main**.
`make prepare-model-release` reads Supabase versions, prompts for major/minor for every non-empty draft, and confirms
before writing local release files and `model-versions.json`. It never deploys or writes Supabase. Both populated
league drafts must be prepared together because they share one training image. Empty drafts retain their prepared
versions. If prediction changes continue after preparation, copy the unpublished release notes back to UNRELEASED.md,
update that consolidated draft and the whole-model explanation, then rerun preparation. It shows the diff and amends
the same unregistered version after confirmation. Never clear a draft manually to bypass deployment or drop prior
changes while consolidating notes. Any nonblank UNRELEASED.md content blocks deployment, including after preparation.
Once registered, new prediction changes require a fresh major/minor bump. Registered notes are immutable.
Preparation rejects stale versions, unrelated existing destinations, and unresolved recovery plans, rechecks registry
versions after confirmation, and restores original files on ordinary write failures. Write notes and versions before
clearing drafts so a process interruption cannot permit deployment under old versions.
After a hard-killed preparation, inspect/restore the affected Git diff before retrying rather than committing partial files.

Production deployments use interactive `make sam-deploy` only. It reads the committed manifest and release notes,
rejects non-empty drafts, downgrades, invalid version increments, and edits to registered notes, prints both versions,
and requires confirmation. The checked-out branch must be `main` and the entire working tree, including untracked
files, must be clean. Check source cleanliness/SHA before build and again before deployment. Before changing AWS,
verify the additive manual-job database columns exist. Recheck registered versions after confirmation and after the
build; a competing release or a planned version registered from different source blocks deployment. Run only one
production deployment at a time; these checks are not a distributed deployment lock. Preparation and deployment show
broad source/dependency review signals against each league's immutable registered SHA, including shared source, notebooks/data, deleted files and
untracked helpers. Changed inputs with a kept registered version require explicit `neutral` classification; `release`
or `abort` stops the command so the draft can be updated. Missing Git history blocks review; a bootstrap release with
no SHA requires manual classification. File changes are indicators, not an automatic major/minor decision or proof of
prediction impact. For a prepared new version, review the changed inputs against its notes and How It Works before
confirming deployment. These signals do not replace recording changes as they happen or guarantee detection of every
semantic change. There is no automatic version bump.
After SAM succeeds, bounded retries verify the API, training, and coordinator Lambdas share the planned Git SHA and
training contains both planned versions. Only then register and verify release rows. Existing canonical release SHAs
remain immutable; initialize a kept bootstrap release's null SHA at most once. Supabase owns deployed timestamps and
canonical SHAs, and a release becomes live when its first successful AWS update sets `first_pick_at`. The tracked
manifest/notes describe prepared intent; new notes must not claim deployment timestamps or source SHAs. Preserve the
historical metadata in existing archives through 2.0.

Deploy and registration recovery never modify tracked files. The ignored `.aws-sam/model-release-plan.json` retains
exact notes, versions, source SHA, and verified deployment time until registry verification succeeds.
Recovery must validate exactly one choice per league, version increments, note fields/hashes, and deployment time
before contacting AWS or writing releases; malformed plans must be restored, not discarded to bypass verification.
Retry a failed build/deploy with `make sam-deploy` on the same clean commit; retain the original plan even if registry
writes already succeeded. Different source/releases cannot overwrite an unfinished plan. If AWS succeeded, `make sam-register-releases`
re-verifies AWS, registers/verifies the saved release snapshots, and clears the plan without rebuilding. Resolve an
old deployment's actual AWS/registry state before deliberately retiring a plan for a source fix. The former
`sam-finalize-release-files` command is removed: preparation replaces post-deployment archiving and the second merge.

On-demand `POST /nfl-update-picks` and `POST /cfb-update-picks` require admin auth and `client-name`, take no body
(or `{}`), and optionally accept an `Idempotency-Key` scoped per league. Reject the reserved interactive client name `notebook`. They use the shared week selector at actual
request time, persist the selected season/week once, and create a one-time schedule at a UTC minute 60–120 seconds
later. A confirmed schedule returns `202` and a run/status URL; no upcoming eligible slate returns `200` /
`not_scheduled`. Old explicit-week bodies return `422`. Keys are echoed on success and `503`; clients needing safe
retries after connection loss should supply their own. Replays never retarget, move, or recreate delivered schedules.
Only still-future `planned` submissions can reconcile an uncertain AWS create. New keys request intentional reruns.

Reuse `scheduled_model_updates` with additive `trigger_source` (`scheduler` default or `api`) and `client_name`
(`aws-scheduler` default). Apply those idempotent setup statements before deployment. Manual rows use `window_key=manual`
and `api:<league>:<key-hash>` run identities. Calendar reconciliation must only read automatic rows. Both sources use
the same atomic claim/completion linkage, while manual events load client/target metadata from their registered row.
A manual job is cancelled before notebook execution if a newer model week has already published. It runs under the
trainer's version/SHA at execution time and never satisfies an automatic plan.

Admin-only `GET /model-update-jobs/{run_key}` returns persisted lifecycle/error fields and the compact linked update
summary, including version/SHA (omit the SHA for historical rows where it is null). Preserve Lambda's log handler and
explicitly enable INFO logging for API/training audit events. Repeated POSTs return the same status (`202` pending,
`200` terminal or unconfirmed).
NFL schedule reads disable caching and honor the requested feed timeout under a process lock; restore nflreadpy
settings afterward. Status reads never mutate or repair jobs. `failed` denotes the last failed attempt, not exhausted AWS retries.
Nonterminal rows more than three hours after `scheduled_for`, and `planned` rows past their delivery time, report
`outcome_unconfirmed=true`; consult logs/failure queues before rerunning. Keep that conservative three-hour bound in
sync with Scheduler/Lambda event-age and runtime settings. No extra failure consumer or progress workflow is present.

Production scheduling is source-controlled in `template.yaml` and `schedule_coordinator.py`. Recurring Scheduler
resources wake the planner at 4:45 AM and noon Eastern; Supabase state suppresses feed work until an active-season
reconciliation or weekly offseason feed check is due. The planner upserts
`scheduled_model_updates` and creates or updates stable, one-time EventBridge schedules. Kickoff changes update the
same named schedule rather than adding another trigger. The current model week cannot advance until its applicable
games are final and the 5:00 AM Eastern rollover following the last game date has arrived. With no future games, the
weekly check creates no training run. Once a future new-season slate is published outside the four-day horizon, the
planner creates one weekly offseason training run; inside four days it switches to active cadence. While active, each
league receives one daily training run at 5:00 AM Eastern through the date of that week's final game. Every game date independently receives up to three
schedule-derived windows: one hour before the first game before 2:00 PM, one hour before the first 2:00-6:59 PM game,
and one hour before the first game at or after 7:00 PM. Empty buckets are skipped; the rule does not depend on weekday.
Completed picks, including a season's final games, are graded only by the next successful prediction run.

EventBridge Scheduler is at-least-once. Stable schedule/run identities, the `scheduled_model_updates` atomic claim,
the serialized training Lambda, and atomic `(model_key, update_id)` completion link jointly prevent duplicate notebook
execution and recover a delivery whose database commit succeeded before its Lambda response completed. The claim lease
is 15 minutes, aligned with the trainer timeout; an active claim must raise rather than acknowledge a retry as successful.
`client_name` remains source metadata (`aws-scheduler` for automatic runs, original client for manual runs); the deterministic `run_key` remains only in the orchestration
path and is not added to model update-history tables. Unscheduled writes do not link to or satisfy a planned run.
One-time schedules delete themselves after delivery, while Supabase retains planned, running, completed, failed,
cancelled, and missed history. Schedule-only changes are infrastructure work and do not populate either
`UNRELEASED.md`, but still deploy via the clean-main release workflow. SAM-local scheduled events remain read-only
because they are not real Lambda runtimes.

## Testing Strategy
1. Run `pytest` for backend smoke coverage.
2. Run `sam local start-api` and hit endpoints using sample events under `events/`.
3. For frontend work, run `cd frontend && npm run typecheck && npm run build`.
4. For notebook-driven NFL or CFB changes, require a human-verified run of the update workflow before merging.
5. Verify persisted pick counts, update history, started-game preservation, graded results when applicable, and the
   corresponding read endpoints.
6. For expected-points recipe work, run `make backtest-expected-points LEAGUE=<nfl|cfb>` for the automatic standard
   deployed-versus-working-tree comparison. `PROFILE=quick` is optional, never a required first step. Treat reports
   as evidence rather than an automatic release gate.

## Security & Compliance
- Never commit secrets; use `.env` locally and secret stores in deployment.
- `template.yaml` should stay parameterized; do not hardcode API keys or DB URLs.
- Backend auth remains API-key based via request headers.
- Supabase access is backend-only through the pooled Postgres connection string in `SUPABASE_DB_URL`.

## Agent Guardrails
- Documentation maintenance is mandatory and part of every change. Whenever behavior, architecture, APIs, schemas,
  environment variables, deployment, testing, or operational workflows change, update `AGENTS.md` and every
  applicable README in the same change by default. Do not wait for the user to request documentation updates.
  Review all affected documentation before finishing, even when the final determination is that no text change is
  necessary, and never knowingly leave documentation stale.
- Never modify large historical datasets under `src/sports/football/cfb/starter_pack_data/` unless explicitly requested.
- Treat notebooks in `src/**/notebook.ipynb` as high-risk for noisy diffs. Prefer Python module edits unless notebook changes are required.
- Do not reintroduce DynamoDB operational persistence or migration tooling unless explicitly requested.
- Keep SQL setup-oriented; assume `db/sql/001_create_sports_models_schema.sql` is the source of truth for Supabase setup.
- Keep model-release SQL idempotent and setup-oriented. Apply the reviewed release-tracking statements to an existing
  Supabase database before the first version-aware production deploy; do not add a migration framework. Preserve its
  single-transaction execution and the `model_version = '1.0'` defaults that keep legacy Lambda writes compatible
  during rollout.
- Do not alter vendored runtime dependencies in `python/` unless the task is specifically about packaging/layer updates.
- For deployment-impacting changes (`template.yaml`, `Dockerfile`, auth logic), require reviewer approval before merge.

## Extensibility Hooks
- Active backend env vars:
  - `LOCALHOST`
  - `ENVIRONMENT`
  - `PORT`
  - `ADMIN_API_KEY`, `FRONT_END_API_KEY`, `READ_API_KEY`, `NBA_API_KEY`, `AWS_API_KEY`
  - `CFBD_API_KEY`
  - `SUPABASE_DB_URL`, `SUPABASE_SCHEMA`
  - `TRAINING_FUNCTION_ARN` (API/coordinator target metadata, supplied by SAM)
  - `SCHEDULER_TARGET_ROLE_ARN`, `SCHEDULE_GROUP_NAME`, `SCHEDULER_DLQ_ARN` (API/coordinator Scheduler resources, supplied by SAM)
- Add new routers in `src/sports/<sport>/<league>/<model>/handler.py` and mount them in `main.py`.
- Add new DB access helpers in `src/utils/db/`.
- Add model-specific Pydantic schemas beside their API boundary; only truly application-wide schemas belong in a
  shared utility package.

## Further Reading
- [`README.md`](README.md)
- [`docs/expected-points-backtesting.md`](docs/expected-points-backtesting.md)
- [`db/sql/001_create_sports_models_schema.sql`](db/sql/001_create_sports_models_schema.sql)
- [`template.yaml`](template.yaml)
- [`main.py`](main.py)
- [`src/sports/football/nfl/expected_points/README.md`](src/sports/football/nfl/expected_points/README.md)
- [`src/sports/football/cfb/expected_points/README.md`](src/sports/football/cfb/expected_points/README.md)
- [`frontend/README.md`](frontend/README.md)

The optional `smoothing_span` adjustment parameter overrides the base EWMA span (default: each metric's existing
10-game span); dynamic metrics still use the larger of that base span and the target week. Research CLI `--span`
sets the same lever. Changing it requires new candidate features and normal comparison evidence.

Explicit CFB training-history studies use `scripts/experiment_cfb_training_history.py` with saved reference/history
frame bundles and a completed reference run. They may add only strictly earlier training rows while preserving every
existing value and the evaluated game population. The standard comparison also permits different histories before the first evaluated season, while regenerating
each recipe's features. It requires identical evaluation populations and identical source outcomes/markets for
shared earlier games, logs training-history differences, and retains original full-history identities in run artifacts.
Each requested start must contain actual usable games; never silently relabel a shorter history. Record source
coverage, exclusions, reconstruction policy, and both full-history fingerprints in the experiment design. See the
canonical backtesting guide's training-history section for parameters and artifact semantics.
