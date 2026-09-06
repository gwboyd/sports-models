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
- `src/utils/db/`: Centralized Postgres data access for operational tables/views.
- `src/utils/`: Shared infrastructure utilities and Pydantic models.
- `frontend/`: Frontend app for serving model views.
- `tests/`: Python test modules for API/Lambda behavior.
- `events/`: Sample AWS SAM event payloads for local invocation.
- `scripts/backtest_expected_points.py`: Read-only NFL/CFB weekly walk-forward runner and baseline comparator.
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
sam local invoke "FastAPILambdaFunction"
sam local start-api
sam local invoke FastAPILambdaFunction -e events/get-health-event.json
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
  JOB --> EP
  API --> NFL[NFL Router]
  API --> CFB[CFB Router]
  API --> NBA[NBA Router]
  NFL --> EP[Shared Expected Points Runtime]
  CFB --> EP
  EP --> NB[Papermill Notebook Execution]
  NB --> DB[(Supabase Postgres)]
  NFL --> DB
  CFB --> DB
  NBA --> DB
  NB --> TMP[/tmp outputs]
  API --> RESP[JSON Responses]
```

`main.py` creates the FastAPI app, validates API keys, and mounts sport/model routers. Operational reads and
writes go through `src/utils/db/sports_models_db.py`. NFL and CFB update endpoints execute their notebooks via the
shared runtime in `src/model_patterns/expected_points/runtime.py`. Shared tracking helpers validate picks, preserve
started games, calculate changes, and grade completed picks. `write_expected_points_run` persists the update record,
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
them. The training Lambda has reserved concurrency of one, and the coordinator and trainer share a verified Git SHA.

NFL expected-points acquisition is isolated in `src/sports/football/nfl/expected_points/data_loader.py`: `nflreadpy`
loads source data as Polars, selects model fields, and converts to pandas at the notebook boundary with caching off.
CFB expected-points uses `src/sports/football/cfb/expected_points/cfbd_client.py` for direct authenticated REST
requests; do not add a conflicting Python SDK dependency.
Opponent-adjusted football efficiency features are implemented in
`src/sports/football/transforms/opponent_adjustment.py`. It fits pre-kickoff ridge offense/defense effects, then
recomputes each team's historical performance against the opponent it faced before applying the established smoother.
Keep downstream feature names stable. Ridge strength, season carryover, CFB's FCS pooling policy, and rating snapshot
cadence are versioned recipe/backtest levers; any change requires regenerated candidate frames and normal baseline
comparison evidence.

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
responses, and the NFL section has nested public methodology and model-insight routes. Approved team logos are local
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
The public NFL methodology lives in `frontend/content/nfl-how-it-works.md` and intentionally preserves the detailed
production Info-page content. Its renderer must continue supporting headings through `h3`, lists, links, inline code,
and responsive images; keep backend and deployment operations in the model README.

Only the deployed AWS training Lambda writes expected-points records automatically; the coordinator writes scheduling
state and plans but never picks, update history, or grading rows. Runtime origin is determined from Lambda runtime
markers while explicitly excluding `AWS_SAM_LOCAL`; `client_name` is audit metadata, not authority.
Local API, `sam local`, and interactive notebook executions are read-only by default. A non-AWS API request must set
`allow_non_aws_write=true`, then it uses the latest registered release version. Notebooks default to
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
and one final refit on all completed games with the selected score parameters. Confidence classifiers train from the
outer out-of-time predictions, tune with their own single chronological split, and refit on that full holdout. Do not
replace this with random splitting or run a second score GridSearch during the final refit.
CFB confidence tuning scores candidate parameters with log loss and always includes the execution-line spread/total
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
`docs/expected-points-backtesting.md`; update it whenever flags, profiles, commands, artifacts, cache behavior, or
metric interpretation changes.

League-level validation belongs in `src/sports/football/<league>/data_validation.py`; generic dataframe contracts
belong in `src/sports/data_validation.py`. Validate only feeds a model actually consumes. Production notebook runs use
`strict_data_validation=True`; warning mode is for feed investigation and must never weaken chronological/leakage
checks. Keep stable loading, validation, feature construction, and training mechanics in Python modules while using
notebooks for orchestration, exploration, charts, and persistence previews.
League-neutral football transforms belong in `src/sports/football/transforms/`; source-specific conversion and feed
shaping belong beside the NFL or CFB workflow. Shared NFL/CFB HTTP schemas belong in
`src/sports/football/expected_points_schemas.py`.

Prediction-affecting expected-points work must be recorded in the applicable
`src/sports/football/<league>/expected_points/UNRELEASED.md`. The file is intentionally empty when there is no
pending model release. A non-empty draft must be released as either a major or minor version by the next production
deployment; it cannot ship under the old version. Drafts use strict Markdown sections for public summary and
changes, with optional evaluation and internal notes. After a successful release, the deployment command archives
the exact draft under `releases/vN.N.md` and resets the working file. Documentation, frontend, API-output, database,
and infrastructure-only changes do not belong in the draft.

Production model deployments use the interactive `make sam-deploy` workflow only. It loads both drafts, prompts for
major/minor for every non-empty draft, keeps empty drafts at their latest version, prints both decisions, and requires
final confirmation. The checked-out branch must be `main`, and the working tree, including untracked files, must be
completely clean so the deployed image matches the recorded Git SHA. NFL and CFB share one training Lambda image, so both populated drafts must be released
in the same deployment. There is no GitHub Actions version gate, source fingerprint, or changed-path heuristic. After
SAM succeeds, the command uses bounded retries to verify that the active training and coordinator Lambdas share the
planned Git SHA and that training contains both planned model versions; only then does it record release rows and
initialize any kept bootstrap release's null source SHA. That conditional initialization is one-time and must never
replace a non-null registry SHA. A release is considered live only after its first successful AWS pick update sets
`first_pick_at`. The ignored `.aws-sam/model-release-plan.json` preserves exact
draft snapshots for recovery. `make sam-register-releases` re-verifies AWS before registering, while
`make sam-finalize-release-files` verifies the Supabase rows before archiving/resetting drafts and removing the plan.

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
execution and recover a delivery whose database commit succeeded before its Lambda response completed. `client_name`
remains source metadata (`aws-scheduler` for these runs); the deterministic `run_key` remains only in the orchestration
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
6. For expected-points recipe work, run a `quick` backtest first and use the `standard` three-season weekly profile for
   normal baseline-versus-candidate evidence. Treat reports as evidence rather than an automatic release gate.

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
  - `TRAINING_FUNCTION_ARN` (coordinator target metadata, supplied by SAM)
  - `SCHEDULER_TARGET_ROLE_ARN`, `SCHEDULE_GROUP_NAME`, `SCHEDULER_DLQ_ARN` (dynamic Scheduler resources, supplied by SAM)
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
