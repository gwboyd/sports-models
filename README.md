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

Version 2.1 keeps the score model unchanged and separates betting probabilities/Lock selection into small shared
heads. Locks target roughly 2–3 combined weekly picks, not a forced quota; stronger candidates can exceed five.
Use `make replay-expected-points-locks LEAGUE=nfl` (or `cfb`) for frozen-score research, with the deployed release as
the default anchor. `LOCK_SOURCE=version:2.0` pins the earlier cache. The
[Lock replay guide](docs/expected-points-backtesting.md#lightweight-lock-method-replay) explains cadence screens,
checksummed calibration-history reuse, and why the final production-path comparison is still required.

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
- `GET /model-update-jobs/{run_key}` (admin only)

For each update, the shared tracking workflow:

1. Validates the predicted pick shape and game IDs.
2. Loads existing picks and results for the selected league.
3. Preserves saved picks for games that have started.
4. Records pick and play changes for update history.
5. Grades previously saved picks when completed scores are available.
6. Atomically persists the update record, picks, and newly graded results.

Only the deployed AWS training Lambda writes automatically. HTTP update routes submit a current-slate refresh through
EventBridge Scheduler; they never train in the HTTP invocation. Local API and SAM-local update requests return `403`
and cannot create production schedules, even when AWS credentials are available. Interactive notebooks remain
read-only by default (`client_name="notebook"`, `allow_non_aws_write=False`). An intentional notebook write requires
`allow_non_aws_write=True` and the exact `WRITE <LEAGUE> <VERSION>` confirmation; it uses the latest registered release.
Explicit season/week and non-AWS write options remain internal notebook-runner inputs, not HTTP request fields.

### On-demand updates

Call either update route with admin `Authorization` and `client-name` headers. The exact client name `notebook` is
reserved for interactive execution and returns `422` on this API. Send no body (an empty `{}` is also
accepted); old season/week or `allow_non_aws_write` bodies are rejected with `422`. The API reads current schedule data
and saved picks, uses the coordinator's model-week selection policy at the actual request time, and saves that exact
season/week. It honors the 5 AM Eastern rollover and can select a published offseason slate. It does not advance the
coordinator's periodic-check state. NFL schedule reads disable nflreadpy caching and use the same five-second feed
timeout as CFB; concurrent NFL fetches serialize briefly so process-global library settings cannot leak between requests.
Feed failures return `503`; no eligible upcoming slate returns `200` with
`status="not_scheduled"` and creates no job.

```shell
curl -i -X POST "$API_URL/nfl-update-picks" \
  -H "Authorization: $ADMIN_API_KEY" \
  -H "client-name: will" \
  -H "Idempotency-Key: nfl-refresh-2026-09-06-1"
```

A confirmed schedule returns `202` with `run_key`, `season`, `week`, `scheduled_for`, and `status_url`; the `Location`
header also points to that status URL. The schedule targets a UTC minute 60–120 seconds after selection, with flexible
windows off; actual training can start later while another run occupies the serialized trainer. The existing
Scheduler role, retry policy, delivery failure queue, and automatic schedule deletion are reused.

`Idempotency-Key` is optional and scoped per league. Reuse it to retrieve the original job without choosing a new week,
changing its client/time, or creating another execution. Omit it or choose a new key for an intentional fresh run.
The effective key is returned in the `Idempotency-Key` response header, including on `503`; supply your own key when
you need safe retries even after a connection loss. Keys accept 1–200 letters, digits, dots, underscores, colons, or
hyphens. An uncertain submission can be retried with that same key before its original delivery time. After that time,
an unconfirmed plan is reported as such and its schedule is never recreated automatically.

```shell
curl "$API_URL/model-update-jobs/$RUN_KEY" -H "Authorization: $ADMIN_API_KEY"
```

GET is read-only and admin-protected for both automatic and manual jobs. It returns persisted status, source/client,
timestamps, attempts, last error, and the linked update ID. Completed jobs include the actual execution version/SHA,
runtime, pick count, and change counts from update history; older rows without a recorded SHA omit that field.
A repeated POST returns `202` while pending, or `200` for a completed/cancelled/missed or unconfirmed job. Missing jobs return `404`.

`failed` means the last attempt failed; AWS may still retry. Pending/failed jobs more than three hours past their
scheduled time have `outcome_unconfirmed=true`; this conservative bound covers the configured Scheduler and Lambda
retry windows. A `planned` row past its delivery time is also unconfirmed. Status reads do not repair jobs or claim
that missing completion proves a failure. Check CloudWatch and the existing failure queues before requesting another
run in that situation. No failure-queue consumer or progress-percentage workflow is added.

Manual plans use `trigger_source="api"`, `window_key="manual"`, and `api:<league>:<key-hash>` identities in the existing
`scheduled_model_updates` table. The calendar coordinator only reconciles `trigger_source="scheduler"` rows. Manual
and automatic jobs share atomic claiming and completion, but cannot satisfy each other's plans. A delayed manual job
is cancelled before training if a newer model week has already been published. Existing kickoff preservation and
grading behavior remains in the notebook workflow.

The trainer records the requesting client and its deployed version/SHA at execution time, including if a deployment
happens after submission. Older preserved picks/results retain their original versions; canonical release SHAs and
`first_pick_at` behavior are unchanged. INFO logging is explicitly enabled for the API/trainer, preserving Lambda's
log handler so request IDs and run keys correlate dispatch/training logs.

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

Both football models tune the score estimator with one predefined chronological validation split inside the earlier
score-training period. A separate outer holdout is strictly later than those training games and supplies genuine
out-of-time score errors and resolved betting outcomes to the probability heads. Version 2.1 fits the residual-based
heads and CFB total base rate on that holdout without classifier GridSearch. After parameter selection and evaluation,
the production score model is refit once on all completed games without repeating GridSearch. Legacy confidence
classifiers retain a chronological validation split and full-holdout refit. This is intentionally a low-cost temporal
evaluation design rather than full out-of-fold backtesting.

Every normal expected-points train also reports a chronology-safe health slice. Team-score, margin, and total
MAE/RMSE/bias come from the outer score holdout. Confidence, calibration, and lock metrics come from a later untouched
portion of that holdout, with historical top-N locks ranked independently inside each season/week. Interactive
notebooks display these metrics, and update rows retain their flat values in the existing `evaluation_metrics` JSON.

For **any prediction-affecting NFL/CFB change**, including adding/removing features or changing training,
run this command from the repository root:

```sh
make backtest-expected-points LEAGUE=nfl
```

Use `LEAGUE=cfb` for CFB; Make defaults to NFL if omitted. This automatically prepares each version's own inputs,
checks compatibility, and compares deployed versus working-tree with the **standard three-season weekly** profile.
No notebook run, separate preparation command, or quick-first run is required. Each version may use different
features and model code while evaluating the same historical games/outcomes/markets. Automatic preparation uses a
conservative completed-season bound; see the guide for the date policy and overrides.

See the [working-tree versus deployed guide](docs/expected-points-backtesting.md#evaluate-working-tree-changes-against-deployed)
for setup, reports, and optional parameters. `PROFILE=quick` requests a smaller preliminary run; it is not a
prerequisite and a subsequent standard run repeats candidate fits with default caching. Other optional controls
include `SEASONS`, `THROUGH_SEASON`, `CADENCE`, `BOOTSTRAP_SAMPLES`, `OUTPUT_DIR`, `BASELINE`, `CANDIDATE`,
`BASELINE_FRAME`, `CANDIDATE_FRAME`, `PREFLIGHT_ONLY=1`, `CACHE_WORKING_TREE=1`, and `NO_CACHE=1`. Advanced
`CURRENT_YEAR`/`CURRENT_WEEK` override preparation context only. Saved frames are used only when explicitly requested;
otherwise each run prepares fresh version-specific inputs. Comparisons retain independent baseline/candidate runs,
input bundles, requests, notebooks, status, and logs. Completed artifacts remain reusable after a later failure.
For repeated opponent-adjustment research on frozen inputs, the backtesting guide also documents an optional
experiment runner; it uses the same fitter and paired reports. The normal command above remains the default.
An explicit [training-history study](docs/expected-points-backtesting.md#explicit-training-history-experiments)
can add older CFB training rows while freezing existing features and evaluated games; it requires saved frame
bundles and checks actual season coverage. The normal command also supports changes to earlier training windows:
it regenerates features, reports differing histories, and requires identical evaluation inputs and shared historical
outcomes/markets. CFB now defaults to 2018 training with 2020 excluded before loading, Elo, and efficiency features.
Generated preparation notebooks preserve and validate the selected source's schema, including legacy release
notebooks; the versioned source notebooks remain untouched.

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
The command prepares differently shaped frames automatically. To reuse known saved inputs, optionally pass
`BASELINE_FRAME=<path>` and/or `CANDIDATE_FRAME=<path>`; compatibility is checked against their shared
game/outcome/market universe rather than requiring identical feature columns. League recipes own final model-frame
assembly and validation. Estimators are never pickled.

Single-run and comparison reports include fixed-seed, season-week block-bootstrap intervals for score/home/away,
margin, total, spread/total result, confidence, calibration, coverage, and lock-frequency metrics. Reports also show
market-implied point-error benchmarks, confidence AUC, mean stated lock probability, and its observed calibration gap.
A lock win rate with no decisions is `N/A`, never zero percent. Notebook-loaded comparison
objects expose `summary`, `deltas`, `baseline_predictions`, `candidate_predictions`, and `lock_changes` as
DataFrames; structured trace fields retain their original list/dict values after cache and artifact round trips.

The runner is read-only with respect to Supabase. It uses currently available historical feed values and explicitly
labels its output as a weekly pre-first-kickoff approximation rather than exact intraday line or roster replay.

The retained CFB classifier feature experiments use the selected execution-line edge and tune with log loss; these
legacy classifiers are not the version 2.1 production probability heads.
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
the first games are graded. NFL and CFB each have a public How It Works route; NFL live model graphics have a
separate Insights route.
The shared presentation uses compact eight-pixel surfaces, limited shadows, and custom electric ink blue (`#0B5FCC`) accents while retaining
44px mobile touch targets. Lock cards use a uniform light-blue outline; favorite cards inherit that outline and show
spread and/or total lock tags when those markets qualify.
If a graded result exists for a current-slate game, green win and red loss outlines replace the blue lock outline for
that individual spread or total everywhere it appears. Standard picks remain unfilled; locks retain their label and
add a light outcome-colored fill. Pushes use a neutral treatment, with outcomes stated in the model details.
Graded current-slate presentations replace kickoff time with the compact away/home final score.
They also show the final margin from the spread pick's team perspective and combined final score for totals while
removing pregame-only card detail.
Both How It Works routes render their public methodology Markdown through a shared renderer, including detailed
feature explanations and responsive images, while developer operations remain in the model READMEs.
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
- one API Lambda for read/serve routes, current-slate update submission, and job status
- one separate training Lambda for direct automatic and manually requested scheduled updates
- one 512 MB schedule-coordinator Lambda that evaluates NFL and CFB calendars
- two recurring planner schedules, one dynamic schedule group, and stack-managed failure queues and IAM roles
- one shared Docker image source built for all three functions

Local `.backtests/` artifacts are excluded from the Docker build context in `.dockerignore`, independently of Git
ignore rules, so generated reports, historical frames, and caches do not enter the Lambda image.

For a model release, prepare its files **on the feature branch before the original commit/merge**:

```shell
make prepare-model-release
```

This reads the latest registered NFL and CFB versions from Supabase, prompts for `major` or `minor` for each non-empty
`UNRELEASED.md`, then asks for confirmation before writing local files. It saves each exact draft under
`src/sports/football/<league>/expected_points/releases/vN.N.md`, empties that draft, and records both intended versions
in the tracked root `model-versions.json`. Preparation does not build, deploy, or write to Supabase. Commit those files
with the model changes and updated public How It Works pages, then merge once. Both leagues share the training image;
prepare both populated drafts together. For documentation, API, infrastructure, or other prediction-neutral changes,
leave the versions unchanged and skip preparation when both drafts are empty. Revising the wording of an existing
unpublished release is an exception: use the amendment workflow below without creating a new version.

`model-versions.json` describes the prepared source, **not whether a release is deployed or live**. Supabase owns the
actual deployment timestamp and canonical source SHA; `first_pick_at` marks the first successful AWS update. New
release Markdown contains no deployment metadata. Existing archives through 2.0 retain their historical metadata.

For any revision to prepared but unpublished notes, including wording-only edits, **reopen `UNRELEASED.md`**: copy the
complete `releases/vN.N.md` into the draft and edit it there. Do not hand-edit the prepared archive or version manifest.
Review How It Works and update it and the local evidence when behavior changes, then run `make prepare-model-release`
again. It shows the note diff
and, after confirmation, amends that same unregistered version and clears the draft. It does not silently create
another version. An empty draft retains an already-prepared release. Once a version is registered in Supabase, new
prediction changes require a fresh major/minor bump; registered notes are immutable. Never manually clear a draft
just to bypass deployment checks or discard earlier release notes while consolidating a later change.

Both preparation and deployment list source/dependency changes relative to each model's immutable registered SHA.
These broad signals include shared source, league code/notebooks/data, runtime configuration, dependencies, deleted
files, and untracked helpers; they do not decide whether predictions changed. If keeping a registered version with
changed inputs, explicitly choose `neutral` only after reviewing them. Choose `release` or `abort` to stop and update
the draft/model documentation. Missing Git history blocks review until fetched; a bootstrap release without a SHA
requires manual classification. A prepared new release still requires reviewing its source changes against its notes
and How It Works. These signals cannot detect the semantic meaning of every change and never replace the requirement
to document prediction changes as they happen, even after preparation.

If the registry has advanced beyond your manifest, update from main and reconcile intent before preparing again.
Preparation never overwrites a registered release or an unrelated existing destination. All choices are validated
before writing, registry versions are rechecked after confirmation, and ordinary write failures restore original
files. Notes and versions are saved before drafts are cleared so an interrupted preparation cannot silently pair
empty drafts with old versions. After a hard-killed preparation, inspect the Git diff and restore the affected local
files before retrying; do not commit a partial preparation.

After review and merge, deploy from a completely clean `main` checkout:

```shell
make sam-deploy
```

That target:

- loads deployment settings from `.env` and requires clean `main`, including no untracked files
- checks that the additive manual-job columns already exist in Supabase before changing AWS
- reads `model-versions.json` and its release notes, rejecting non-empty drafts, missing/edited registered notes,
  downgrades, and versions other than the next major or minor
- prints source-review signals and both prepared versions, requires any applicable prediction-neutral classification,
  and asks for deployment confirmation; major/minor selection happens only during preparation
- rechecks registered versions after confirmation and after `sam build`, rejecting competing releases or a planned
  version registered from different source; verifies the source is still clean and unchanged, then deploys
  `sports-models-v2` in `us-east-1`
- verifies the active API, training, and coordinator Lambdas share the planned Git SHA and target configuration,
  and training carries both planned model versions, using bounded retries for AWS propagation
- registers new release rows only after AWS verification; existing releases keep their original canonical SHA
  (a kept bootstrap release's null SHA is initialized exactly once)
- verifies registration and removes the ignored recovery plan; **no tracked files are modified**

There is no post-deployment documentation commit or second merge. Redeploying the same prepared versions for
prediction-neutral changes uses the same command. Previously locked picks retain their original model versions.
Direct `sam deploy` bypasses this workflow and is unsupported for production. Run only one production deployment
at a time; the registry checks do not provide a distributed deployment lock.

The confirmed note snapshots, versions, Git SHA, and verified deployment time are retained in ignored
`.aws-sam/model-release-plan.json` until registration is verified. If build or deployment fails, fix the external
cause and rerun `make sam-deploy` at the same clean commit; it resumes that plan without replacing its snapshots.
A plan for different source/releases blocks a new deployment. If a source fix is necessary, first inspect AWS and
resolve the old deployment/registration state before deliberately retiring the old local plan; never discard it
merely because a run was quiet.

If AWS deployment succeeded but verification or registration failed, recover without rebuilding:

```shell
make sam-register-releases
```

Recovery first validates that the plan contains exactly one choice per league, valid version increments, matching
note fields/hashes, and a valid deployment timestamp when present. Damaged plans fail before any AWS or registry work;
restore the original plan instead of discarding it to bypass verification. It rechecks registry compatibility and all
three Lambdas against the saved plan, registers/verifies the exact release rows, and clears
that plan without modifying tracked files. It uses saved notes even if your checkout has since changed. A failed
AWS verification cannot register a release. The former `sam-finalize-release-files` command is no longer needed or
provided; there is no post-deployment archive step.

Release notes flow from `UNRELEASED.md` through the prepared `releases/vN.N.md` into
`sports_models.model_releases`: the title maps to `title`, `## Public Summary` to `public_summary`, and `## Changes`
to `changes_md`. Use a plain-language summary and 2–3 concise change bullets explaining meaningful implementation
details, such as estimator family, chronological training, shrinkage, or selection thresholds. Optionally add one
evaluation-versus-baseline bullet under `## Changes`, naming the baseline, period/population, assumptions, and material
uncertainty or selection limitations. Use verified results for the actual recipe; omit the bullet if those caveats
cannot fit responsibly. Avoid file paths, commands, infrastructure details, and experiment ledgers. Omitted optional
headings become null `evaluation_md`/`internal_notes_md`; do not add separate evaluation/internal headings to new drafts.
Keep complete evaluation evidence in Git-ignored
`.backtests/expected_points/reports/`. The manifest selects versions; the Markdown title does not. Deployment supplies
the model key, major/minor numbers, verified source SHA, and deployment timestamp; no manual release-row SQL is needed.

### Scheduled Pick Updates

`template.yaml` owns the production EventBridge Scheduler configuration. Schedule changes therefore follow the same
clean-`main` `make sam-deploy` workflow as other infrastructure changes, but they do not require an expected-points
version bump or an `UNRELEASED.md` entry unless the model recipe also changes.

EventBridge invokes the small coordinator at 4:45 AM and noon Eastern. The early invocation registers the separate
5:00 AM training schedule after evaluating the 5:00 AM rollover gate. The coordinator reads
`schedule_coordinator_state` first; during the offseason it loads schedule feeds only for the weekly check. Apply the idempotent `schedule_coordinator_state` and
`scheduled_model_updates` statements in `db/sql/001_create_sports_models_schema.sql` to an existing Supabase database
before deploying this stack version. In particular, apply both additive `trigger_source` and `client_name` column
statements to existing run tables before deploying the on-demand API. Existing rows default to automatic scheduling.
Validate with `sam validate --lint --template-file template.yaml` and `pytest`; then use the clean-main
`make sam-deploy` workflow. After deployment, human-verify one keyed update, its replay and GET status, persisted
pick/update counts, version attribution, started-game preservation, applicable grading, and the read endpoints.

When active, the coordinator loads nflverse and CFBD schedules, persists the upcoming plan in Supabase, and creates or
updates named one-time EventBridge schedules. Those schedules—not the coordinator—send IAM-authorized events directly
to the training Lambda. Neither path calls API Gateway or carries an API key. If a kickoff moves, the stable schedule
name and `aws-scheduler:<league>:...` run identity cause both AWS and Supabase to be updated instead of duplicated.

Week transition is backend-gated. The coordinator keeps the currently published week active until every applicable
schedule game is final and 5:00 AM Eastern has arrived on the morning after its last game date. It then opens the next
same-season week. A new season switches to active cadence when its first week enters the four-day horizon. The
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
identifies automatic runs; API-created schedules retain the original requesting client. Unscheduled notebook writes
have no schedule link and cannot satisfy a planned run. Completed duplicates are no-ops, including recovery when
persistence committed before the Lambda response completed; failed runs release the claim for Lambda retry. Claims
use a 15-minute lease matching the serialized trainer's timeout. An unexpired running claim raises an error rather than acknowledging a retry as
successful; terminal duplicates remain no-ops. One-time AWS schedules delete themselves after delivery, while
Supabase retains their lifecycle for operations and a future frontend view. Sample coordinator and direct-training
events are available at `events/schedule-coordinator.json`, `events/scheduled-nfl-update.json`, and
`events/scheduled-cfb-update.json`. SAM-local scheduled invocations remain read-only under the normal non-AWS write
policy; only deployed Lambda runtimes write automatically.

For each model, an empty `UNRELEASED.md` means no unprepared changes; a new version can already be prepared in
`model-versions.json` and `releases/vN.N.md`. A non-empty draft uses the documented Markdown sections and either prepares
the next minor/major release or amends the same unpublished prepared version. Version rows may be deployed before they
make picks; `first_pick_at` is set by the first successful AWS update and is the live-state marker. Existing records
are attributed to the `1.0` bootstrap baseline.

After deploy, verify the API:

```shell
aws cloudformation describe-stacks --region us-east-1 --stack-name sports-models-v2
curl https://your-api-id.execute-api.us-east-1.amazonaws.com/health
```

## Documentation Maintenance

Treat model improvement as a complete loop: define the hypothesis and evaluation population, implement with
chronology checks, compare against deployed with the standard command, record variants and uncertainty in local
studies/reports, then update the release notes and complete public methodology before review/deployment. If a user
waives reevaluation after a prediction change, record that gap and do not reuse earlier gains as proof of the new recipe.

Model releases have three documentation surfaces. `UNRELEASED.md` is a short public account of what changes;
`frontend/content/nfl-how-it-works.md` and `frontend/content/cfb-how-it-works.md` explain how the applicable model
version operates, including its assumptions and limitations. They must describe the whole current model rather than emphasizing only the latest release. Review those pages
whenever a version changes and update both when shared behavior changes. Coordinate frontend publication with the backend release, and keep
version labels, metrics, history windows, confidence rules, and scheduling accurate.

Model improvement, experiment, and optimization reports belong in Git-ignored `.backtests/expected_points/reports/`.
Code-review findings and routine implementation/status summaries belong in chat unless a saved document is requested.
Individual evaluation jobs
belong in `comparisons/<league>/<timestamp>/`, while studies across variants belong in `experiments/<dated-study>/`
and reference their comparison jobs. A local `.backtests/expected_points/README.md` indexes selected evidence.
The directory-specific `.backtests/expected_points/AGENTS.md` explains how to read artifacts and write the
study plans, ledgers, and reports that runners do not create automatically; both local guides are Git-ignored.
The [backtesting guide](docs/expected-points-backtesting.md#saved-results-and-recovery) documents the complete layout
and cleanup policy. They are local-only evidence, not public release notes or tracked
documentation. Do not force-add them. Share an explicit export outside Git when needed.


Documentation is part of the implementation. Any change to behavior, architecture, APIs, schemas, environment
variables, deployment, testing, or operational workflows must update `AGENTS.md` and every applicable README in
the same change. Contributors and coding agents should perform this documentation review by default rather than
waiting for a separate request.
