# sports-models

Backend services for sports prediction models, deployed on AWS Lambda with SAM and backed by Supabase Postgres.

## Overview

- FastAPI app served locally with `uvicorn` and in AWS through Lambda + API Gateway
- Supabase Postgres for persisted picks and results data
- Separate SAM-managed Lambdas for API traffic and long-running NFL/CFB training updates

The operational database schema is `sports_models`.

## Setup

Use Python `3.11.x` to match the Lambda container runtime.

Install `uv`, then create and activate the Python version pinned by `.python-version`:

```shell
brew install uv
uv venv
source .venv/bin/activate
```

Install system dependencies:

```shell
brew install libomp
brew install python-certifi
```

Install runtime and local test dependencies:

```shell
uv pip install -r requirements-dev.txt
uv pip check
```

`requirements.txt` is the Lambda/runtime dependency set. `requirements-dev.txt` includes it and adds pinned local
test dependencies; use the dev file for a reproducible contributor environment.

Create the backend env file:

```shell
cp .env.example .env
```

Required values in `.env`:

- `ADMIN_API_KEY`
- `FRONT_END_API_KEY`
- `READ_API_KEY`
- `NBA_API_KEY`
- `AWS_API_KEY`
- `CFBD_API_KEY`
- `SUPABASE_DB_URL`
- `SUPABASE_SCHEMA`

For local development, also set:

- `LOCALHOST=True`
- `SUPABASE_SCHEMA=sports_models`

Use the pooled Supabase Postgres connection string for `SUPABASE_DB_URL`.

## Database

For a fresh or explicitly reviewed compatible database, run:

```text
db/sql/001_create_sports_models_schema.sql
```

This creates the `sports_models` schema and the tables/views used by the backend:

- `nfl_expected_points_picks`
- `nfl_expected_points_latest_picks`
- `nfl_expected_points_results`
- `nfl_expected_points_pick_updates`
- `nfl_expected_points_latest_updates`
- `cfb_expected_points_picks`
- `cfb_expected_points_latest_picks`
- `cfb_expected_points_results`
- `cfb_expected_points_pick_updates`
- `cfb_expected_points_latest_updates`
- `model_releases`
- `schedule_coordinator_state`
- `scheduled_model_updates`
- `nba_first_basket_picks`

Do not treat the setup file as an incremental production migration without reviewing its statements against
the deployed schema. For an existing database, apply only the scoped, reviewed SQL required by the change.

Expected-points reads and writes are league-aware and centralized in
`src/utils/db/sports_models_db.py`. An update writes its run-history record, current picks, and any newly graded
results in one transaction. If any part fails, the transaction rolls back instead of leaving a partial run.
Initial Postgres connection failures receive a small number of bounded retries.

CFB pick and result tables also store nullable `home_conference` and `away_conference` metadata. Existing Supabase
environments created before those fields were introduced require the scoped `alter table ... add column if not
exists` statements in the setup SQL before deploying a backend that reads or writes the fields.

Expected-points release tracking adds `model_version` to NFL/CFB picks, results, and update history, plus the shared
`model_releases` catalog. Locked picks and later grading retain the recipe version that produced the pick, even when a
newer deployment is live. Apply the idempotent release-tracking statements in the setup SQL to an existing Supabase
database before the first version-aware deployment. The setup seeds a `1.0` baseline and attributes existing history
to it; the corresponding `releases/v1.0.md` files document the production recipes present when tracking began while
making clear that exact pre-versioning revisions cannot be reconstructed. Future production writes reject an
unregistered version. The setup runs in one transaction and assigns a
`1.0` column default before enforcing `NOT NULL`, so the pre-versioning Lambda can continue writing safely during the
schema-to-code rollout. Run it when no model update is active to avoid waiting on its brief table/index locks.

## Expected Points Workflows

NFL and CFB use the shared modeling, tracking, reporting, notebook-execution, and persistence helpers under
`src/model_patterns/expected_points/`. Sport-specific notebooks remain responsible for producing the model input
and predictions. League-neutral football feature transforms live under `src/sports/football/transforms/`; feed and
league adapters remain beside their NFL or CFB workflow. Shared HTTP schemas for the two expected-points APIs live
in `src/sports/football/expected_points_schemas.py` rather than the application-wide utility layer.

NFL ingestion is isolated in `src/sports/football/nfl/expected_points/data_loader.py`. It uses `nflreadpy` for
play-by-play, weekly player statistics, schedules, and team metadata; selected Polars frames are converted to pandas
at that boundary so model feature engineering remains pandas-based. The loader disables the client cache, preserves
legacy float downcasting and team-abbreviation normalization, and only tolerates an unavailable current-season
release during week 1. CFB data uses the direct REST client in
`src/sports/football/cfb/expected_points/cfbd_client.py`, rather than a Python SDK. When CFBD adds a new game field,
the notebook retains that API schema and represents the field as null for older historical CSV rows; the same
alignment applies to advanced game-stat fields. Scheduled games retain their pregame Elo and never update team
ratings until both final scores are available. Score-model training coerces final scores to numeric values and excludes
any game missing or invalid for either target.

Available routes:

- `GET /nfl-picks`
- `GET /nfl-pick-results`
- `POST /nfl-update-picks`
- `GET /cfb-picks`
- `GET /cfb-pick-results`
- `POST /cfb-update-picks`

For each update, the shared tracking workflow:

1. Validates the predicted pick shape and game IDs.
2. Loads existing picks and results for the selected league.
3. Preserves saved picks for games that have started.
4. Records pick and play changes for update history.
5. Grades previously saved picks when completed scores are available.
6. Atomically persists the update record, picks, and newly graded results.

Only the deployed AWS training Lambda writes automatically. Local servers, `sam local`, and other non-AWS runtimes
are read-only unless the update request explicitly sets `allow_non_aws_write=true`; an authorized non-AWS run uses
the latest registered release version. Interactive notebooks default to `client_name="notebook"` and
`allow_non_aws_write=False`. Enabling the flag in a notebook requires typing `WRITE <LEAGUE> <VERSION>` before the
shared transaction writer can persist anything. Manual update rows retain the notebook/client name and local Git
identity for auditing.

CFB market selection is deterministic and independent of CFBD provider ordering. A game enters the model when at
least one participant is FBS and at least one real sportsbook supplies each of the current spread and total. A
moneyline, opening line, or second provider is not required. Synthetic CFBD sources such as `consensus`,
`numberfire`, and `teamrankings` are not treated as executable sportsbook quotes.

The CFB score model remains market-aware through implied team totals calculated from median sportsbook reference
lines. The model chooses its spread and total directions against those references, then shops the execution quote
within the largest provider cluster whose full range is no more than two points. Tied clusters prefer the one nearest
the overall median, then the more reliable provider mix. When no two books corroborate one another, the workflow uses
the quote nearest consensus without shopping. Exact provider ties use Bovada, William Hill, ESPN Bet, DraftKings,
Caesars, then remaining providers alphabetically. A one-book market uses its sole quote. Unsupported markets remain
visible and trainable but are forced to non-lock status. The selected execution quote—not the reference—drives the
model edge, confidence, persisted/displayed line, and eventual grading. Spread and total support are independent.

CFBD does not expose spread/total prices or provider-update timestamps, so "best line" means the most favorable
corroborated point for the direction already selected, not the best expected payout. Provider, reference/opening
lines, quote counts/ranges, support status, and selection reasons are retained in the existing pick-update JSON; no
separate market-snapshot table is used. Games at or after kickoff are removed before prediction, even when no earlier
pick exists.

Both football models tune the score estimator with one predefined chronological validation split inside an outer
chronological holdout. The outer holdout is strictly later than the score-model training games and supplies genuine
out-of-time predictions to the confidence classifiers. After parameter selection and evaluation, the production
score model is refit on all completed games without running a second GridSearch; each confidence classifier likewise
uses one chronological validation split and then refits on its full outer-holdout dataset. This is intentionally a
low-cost temporal evaluation design rather than full out-of-fold backtesting.

Every normal expected-points train also reports a chronology-safe health slice. Team-score, margin, and total
MAE/RMSE/bias come from the outer score holdout. Confidence, calibration, and lock metrics come from a later untouched
portion of that holdout, with historical top-N locks ranked independently inside each season/week. Interactive
notebooks display these metrics, and update rows retain their flat values in the existing `evaluation_metrics` JSON.

The shared walk-forward runner provides fuller baseline-versus-candidate evidence by retraining an unfitted NFL or
CFB recipe before historical weekly slates. In either notebook, set `write_backtest_frame=True` and run through the
dedicated frame-save stage; this does not require running the normal model train. Then run:

```sh
make backtest-expected-points LEAGUE=nfl PROFILE=standard BASELINE=deployed
```

See [`docs/expected-points-backtesting.md`](docs/expected-points-backtesting.md) for the complete notebook, CLI,
baseline-resolution, cache, artifact, and interpretation workflow. The Make target also accepts `SEASONS`, `CADENCE`,
`BOOTSTRAP_SAMPLES`, `OUTPUT_DIR`, `BASELINE_FRAME`, `CANDIDATE_FRAME`, `CACHE_WORKING_TREE=1`, and `NO_CACHE=1`.

`quick` samples the latest evaluable season, `standard` runs every week across the latest three, and `full` requests
four. `BASELINE` accepts `deployed`, `version:N.N`, `sha:<git-sha>`, or `artifact:<path>`. Historical refs execute in
isolated Git worktrees; dependency changes use fingerprinted environments cached under `.backtests/expected_points/`.
A ref predating this protocol needs a bootstrapped run artifact. Deployed/version resolution uses the immutable
`source_git_sha` registered when that live model version was released. Later retraining runs under the same version do
not move its baseline. The bootstrap 1.0 rows predate SHA tracking, so the first verified protocol-capable deployment
fills their currently missing registry SHA exactly once. Per-run pick-update SHAs remain audit metadata and are not
used for baseline resolution. Recipe identities include the Git SHA, relevant
dirty/untracked Python content discovered recursively rather than through a filename allowlist, recipe
protocol/configuration, and dependency fingerprint. Ignored local data, cutoff
caches, predictions, and reports live under `.backtests/expected_points/`. New weeks append to a compatible cache;
an earlier feed correction invalidates that cutoff and every later expanding-history fit. Working-tree cutoffs use a
single rolling namespace per league to avoid accumulating a directory for every local edit. `deployed` and
`version:N.N` share a readable `version-N.N` namespace, with full metadata validation before reuse; explicit Git SHA
references retain independent fingerprinted namespaces.

Working-tree cutoff caching is disabled by default, so local candidate comparisons are fresh fits and write no cutoff
cache. Set `CACHE_WORKING_TREE=1` (or notebook parameter `backtest_cache_working_tree=True`) to opt into resumable
rolling caching for an expensive unchanged candidate. Released/version/SHA baselines continue caching by default.

The saved frame is input data, not a cached model or substitute for recipe code: every cutoff refits from scratch.
When recipe versions require differently prepared frames, pass `BASELINE_FRAME=<path>` and
`CANDIDATE_FRAME=<path>`; compatibility is checked against their shared game/outcome/market universe rather than
requiring identical feature columns. League recipes own final model-frame assembly and validation. Estimators are
never pickled.

Single-run and comparison reports include fixed-seed, season-week block-bootstrap intervals for score/home/away,
margin, total, spread/total result, confidence, calibration, coverage, and lock-frequency metrics. Reports also show
market-implied point-error benchmarks, confidence AUC, mean stated lock probability, and its observed calibration gap.
A lock win rate with no decisions is `N/A`, never zero percent. Notebook-loaded comparison
objects expose `summary`, `deltas`, `baseline_predictions`, `candidate_predictions`, and `lock_changes` as
DataFrames; structured trace fields retain their original list/dict values after cache and artifact round trips.

The runner is read-only with respect to Supabase. It uses currently available historical feed values and explicitly
labels its output as a weekly pre-first-kickoff approximation rather than exact intraday line or roster replay.

CFB confidence classifiers always receive the selected execution-line edge and select hyperparameters with log loss.
Raw home moneyline is not an eligibility gate; paired provider moneylines are converted to an optional median no-vig
home probability with an explicit missing indicator. Opening-line, movement, market-depth, and no-vig feature groups
must pass a genuine out-of-time gate before production use: pooled Brier score improves by at least one percent, log
loss does not worsen, and no evaluated season's Brier score worsens by more than two percent. Read-only CFB notebook
runs evaluate these groups on a later untouched slice of the score model's outer holdout and report Brier score, log
loss, and calibration error; API-triggered runs skip this analysis. No optional group is currently promoted.

Shared structural data contracts check required columns, nonempty feeds, requested season coverage, stable keys, and
assembled model-frame uniqueness. NFL and CFB notebooks expose `strict_data_validation=True`: strict mode raises on
contract drift, while `False` logs warnings for feed investigation. Leakage-prevention checks—chronological splits
and the ordering of lagged features—always fail when their time ordering cannot be proved.

CFB advanced metrics are placed on the complete team-game schedule before lagging. Completed rows contribute the raw
observations; scheduled rows receive an EWMA derived only from strictly earlier kickoffs. A current prediction row
whose monitored efficiency features are all null is a contract failure instead of silently becoming a median-only
prediction.

The supplemental NFL power-ranking classifier also uses a predefined train-before-validation split. It is used only
for chart generation and never supplies pick or confidence features.

The mobile-first NFL and CFB frontend presents favorites, separate spread/total lock cards, and a single game-centered
slate. CFB games appear once and can be filtered by either team's conference. Shared results routes derive season and
weekly summaries from the existing graded-game responses; an unavailable CFB result set renders an empty state until
the first games are graded. NFL methodology and live model graphics are available on separate nested routes.
The shared presentation uses compact eight-pixel surfaces, limited shadows, and custom electric ink blue (`#0B5FCC`) accents while retaining
44px mobile touch targets. Lock cards use a uniform light-blue outline; favorite cards inherit that outline and show
spread and/or total lock tags when those markets qualify.
If a graded result exists for a current-slate game, green win and red loss outlines replace the blue lock outline for
that individual spread or total everywhere it appears. Standard picks remain unfilled; locks retain their label and
add a light outcome-colored fill. Pushes use a neutral treatment, with outcomes stated in the model details.
Graded current-slate presentations replace kickoff time with the compact away/home final score.
The NFL How It Works route renders the full public methodology document, including its detailed feature sections and
responsive Markdown charts, while developer operations remain in the model README.
On mobile, the lock carousel begins on the page content line, and favorite-team management is reached through the
Favorites section rather than a duplicate hero action. Search and favorite sheets follow the visible browser viewport
and freeze background scrolling so iOS software-keyboard changes do not move the sheet off screen.
Kickoff labels are localized to the viewer's device timezone in the browser. Both NFL and CFB `date_time` values use
`America/New_York` wall time with daylight-saving transitions; raw CFBD UTC timestamps are converted at preparation.
Model-update timestamps are also localized and display the device timezone's current seasonal abbreviation.
The product header is branded as Boyd's Picks and temporarily exposes only NFL and CFB navigation; the direct NBA
route remains available. Results summaries place lock records first and omit the redundant predicted-games tile.
Favorite and mobile game cards emphasize actionable picks, with blue outlines applied to individual locked markets.

Football team metadata and local logo assets can be refreshed without a frontend API change. `make sync-cfb-teams
YEAR=2026` loads `CFBD_API_KEY` from the root `.env`, writes a deterministic CFB manifest for the selected FBS season,
and caches its logos under `frontend/public/teams/cfb/`. `make sync-nfl-teams` mirrors the nflverse-data teams release
used by the NFL loader. `make sync-football-teams YEAR=2026` refreshes both catalogs.

## Local Development

Run the backend directly:

```shell
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 3000 --reload
```

Or use the shortcut:

```shell
make backend
```

Health check:

```shell
curl http://127.0.0.1:3000/health
```

## Testing

Backend tests:

```shell
pytest
```

When validating a dependency removal or Lambda-compatible wheel set, create a fresh Python 3.11 virtual environment
instead of installing over an existing `.venv`, then run `uv pip check` after installing `requirements-dev.txt`. The
NFL loader tests cover the Polars-to-pandas compatibility boundary; live notebook updates still require the
operational checks below.

Notebook-driven NFL or CFB workflow changes also require a human-verified update run before merging. Confirm the
pick count, update-history row, started-game preservation, graded results when applicable, and the corresponding
read endpoints.

SAM local build:

```shell
sam build
```

Or:

```shell
make sam-build
```

Invoke the sample health event locally:

```shell
make sam-invoke-health
```

Run the local SAM API:

```shell
make sam-api
```

That serves the API at `http://127.0.0.1:3001`.

Frontend local testing notes live in [frontend/README.md](/Users/willboyd/Desktop/Repos/sports-models/frontend/README.md).

## Deploy

The SAM template deploys:

- one HTTP API
- one API Lambda for read/serve routes
- one separate training Lambda for `POST /nfl-update-picks`, `POST /cfb-update-picks`, and direct scheduled updates
- one 512 MB schedule-coordinator Lambda that evaluates NFL and CFB calendars
- two recurring planner schedules, one dynamic schedule group, and stack-managed failure queues and IAM roles
- one shared Docker image source built for all three functions

Repeat production deploys:

```shell
make sam-deploy
```

That target:

- loads deploy settings from `.env`
- requires a completely clean Git working tree, including no untracked files
- requires the checked-out branch to be `main`
- loads the latest NFL and CFB release rows from Supabase
- reads each model's `UNRELEASED.md` queue
- requires a major/minor choice for every non-empty queue and keeps empty queues unchanged
- prints both model decisions and requires confirmation
- runs `sam build` and deploys the `sports-models-v2` stack in `us-east-1`
- verifies the deployed training and coordinator Lambdas are active, share the planned Git SHA, and have the planned
  model versions and target function configuration, using bounded retries for AWS propagation
- registers finalized release rows only after SAM succeeds; for a kept bootstrap release whose registry SHA is still
  null, initializes that SHA exactly once from the verified deployment; then archives and resets consumed drafts

NFL and CFB share the training Lambda image. A populated queue for either model therefore ships in the same AWS
deployment and must receive a release decision. Direct `sam deploy` bypasses this safety workflow and is not supported
for production. The confirmed draft snapshots are retained locally in the ignored
`.aws-sam/model-release-plan.json` until the entire workflow completes. If SAM succeeds but AWS verification or
release registration fails, retry registration with the first command below; it rechecks the actual Lambda before
writing Supabase. If local archive/reset fails, use the second command; it verifies the exact Supabase rows before
changing Markdown:

```shell
make sam-register-releases
make sam-finalize-release-files
```

Successful finalization removes the recovery plan. A failed SAM deployment cannot be registered unless a later AWS
check confirms that the training and coordinator Lambdas have the planned configuration and Git SHA.

### Scheduled Pick Updates

`template.yaml` owns the production EventBridge Scheduler configuration. Schedule changes therefore follow the same
clean-`main` `make sam-deploy` workflow as other infrastructure changes, but they do not require an expected-points
version bump or an `UNRELEASED.md` entry unless the model recipe also changes.

EventBridge invokes the small coordinator at 4:45 AM and noon Eastern. The early invocation registers the separate
5:00 AM training schedule after evaluating the 5:00 AM rollover gate. The coordinator reads
`schedule_coordinator_state` first; during the offseason it loads schedule feeds only for the weekly check. Apply the idempotent `schedule_coordinator_state` and
`scheduled_model_updates` statements in `db/sql/001_create_sports_models_schema.sql` to an existing Supabase database
before deploying this stack version.

When active, the coordinator loads nflverse and CFBD schedules, persists the upcoming plan in Supabase, and creates or
updates named one-time EventBridge schedules. Those schedules—not the coordinator—send IAM-authorized events directly
to the training Lambda. Neither path calls API Gateway or carries an API key. If a kickoff moves, the stable schedule
name and `aws-scheduler:<league>:...` run identity cause both AWS and Supabase to be updated instead of duplicated.

Week transition is backend-gated. The coordinator keeps the currently published week active until every applicable
schedule game is final and 5:00 AM Eastern has arrived on the morning after its last game date. It then opens the next
same-season week. A new season cannot open until its first week enters the four-day horizon. The
next successful notebook run grades prior picks and atomically publishes its slate. If no future games have been
published, the weekly coordinator check creates no training run. Once the next season's games exist but its first
kickoff is more than four days away, one weekly offseason training run is scheduled; this also gives the normal
notebook workflow a chance to grade the prior season's final picks. Four days before kickoff, active cadence replaces
the weekly offseason run. The coordinator itself never writes picks, update history, or grading rows.

While a league has an eligible future week, it receives one training update at 5:00 AM Eastern every day through that
week's final game date. Every date containing games also receives up to three window updates: one hour before the
first game before 2:00 PM, one hour before the first 2:00-6:59 PM game, and one hour before the first game at or after
7:00 PM. Empty buckets are skipped, and the policy is identical on every weekday. Two games in the same window share
one update before the earlier kickoff; games spread across Saturday and Sunday can create three windows on each day.
Schedules use the exact kickoff minute: a 7:10 PM first game schedules at 6:10 PM.

The training Lambda has reserved concurrency of one, allowing simultaneous NFL/CFB decisions to serialize safely.
Because EventBridge Scheduler delivers at least once, the training handler atomically claims the Supabase plan before
Papermill starts. A successful database transaction stores the new model-specific update row and writes its `id` to
the plan's generic `update_id` while marking the plan completed. `model_key` identifies which update-history table to
join, so adding a scheduled model does not require another link column. This polymorphic link is written atomically
rather than declared as a cross-table foreign key, which Postgres cannot express. `client_name='aws-scheduler'`
continues to identify the writer in update history; unscheduled writes have no schedule link and cannot satisfy a
planned run. Duplicate or concurrent deliveries are no-ops, including recovery when persistence committed before the
Lambda response completed; failed runs release the claim for Lambda retry. One-time AWS schedules delete themselves after delivery, while
Supabase retains their lifecycle for operations and a future frontend view. Sample coordinator and direct-training
events are available at `events/schedule-coordinator.json`, `events/scheduled-nfl-update.json`, and
`events/scheduled-cfb-update.json`. SAM-local scheduled invocations remain read-only under the normal non-AWS write
policy; only deployed Lambda runtimes write automatically.

For each model, an empty `UNRELEASED.md` means no version change. A non-empty file must use the documented Markdown
sections and is released as either the next minor or next major version. Version rows may be deployed before they
make picks; `first_pick_at` is set by the first successful AWS update and is the live-state marker. Existing records
are attributed to the `1.0` bootstrap baseline.

After deploy, verify the API:

```shell
aws cloudformation describe-stacks --region us-east-1 --stack-name sports-models-v2
curl https://your-api-id.execute-api.us-east-1.amazonaws.com/health
```

## Documentation Maintenance

Documentation is part of the implementation. Any change to behavior, architecture, APIs, schemas, environment
variables, deployment, testing, or operational workflows must update `AGENTS.md` and every applicable README in
the same change. Contributors and coding agents should perform this documentation review by default rather than
waiting for a separate request.
