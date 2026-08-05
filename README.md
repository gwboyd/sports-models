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

## Expected Points Workflows

NFL and CFB use the shared modeling, tracking, reporting, notebook-execution, and persistence helpers under
`src/model_patterns/expected_points/`. Sport-specific notebooks remain responsible for producing the model input
and predictions.

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

Interactive notebook executions use `client_name="notebook"` and remain read-only. API-triggered executions use a
non-notebook client name and persist through the shared transaction writer.

CFB games must have a selected betting provider and home moneyline to reach the expected-points model. Games that
exist in the CFBD schedule but do not yet have the required market data are excluded from the current prediction
frame. Games at or after kickoff are also removed before prediction, even when no earlier pick exists.

Both football models tune the score estimator with one predefined chronological validation split inside an outer
chronological holdout. The outer holdout is strictly later than the score-model training games and supplies genuine
out-of-time predictions to the confidence classifiers. After parameter selection and evaluation, the production
score model is refit on all completed games without running a second GridSearch; each confidence classifier likewise
uses one chronological validation split and then refits on its full outer-holdout dataset. This is intentionally a
low-cost temporal evaluation design rather than full out-of-fold backtesting.

Shared structural data contracts check required columns, nonempty feeds, requested season coverage, stable keys, and
assembled model-frame uniqueness. NFL and CFB notebooks expose `strict_data_validation=True`: strict mode raises on
contract drift, while `False` logs warnings for feed investigation. Leakage-prevention checks—chronological splits
and the ordering of lagged features—always fail when their time ordering cannot be proved.

CFB advanced metrics are placed on the complete team-game schedule before lagging. Completed rows contribute the raw
observations; scheduled rows receive an EWMA derived only from strictly earlier kickoffs. A current prediction row
whose monitored efficiency features are all null is a contract failure instead of silently becoming a median-only
prediction.

The mobile-first NFL and CFB frontend presents favorites, separate spread/total lock cards, and a single game-centered
slate. CFB games appear once and can be filtered by either team's conference. Shared results routes derive season and
weekly summaries from the existing graded-game responses; an unavailable CFB result set renders an empty state until
the first games are graded. NFL methodology and live model graphics are available on separate nested routes.
The shared presentation uses compact eight-pixel surfaces, limited shadows, and custom electric ink blue (`#0B5FCC`) accents while retaining
44px mobile touch targets. Lock cards use a uniform light-blue outline; favorite cards inherit that outline and show
spread and/or total lock tags when those markets qualify.
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
- one separate training Lambda for `POST /nfl-update-picks` and `POST /cfb-update-picks`
- one shared Docker image build used by both functions

Repeat deploys:

```shell
make sam-deploy
```

That target:

- loads deploy settings from `.env`
- runs `sam build`
- deploys the `sports-models-v2` stack in `us-east-1`

If you prefer the raw commands:

```shell
set -a
source .env
set +a
sam build
sam deploy \
  --stack-name sports-models-v2 \
  --region us-east-1 \
  --resolve-s3 \
  --resolve-image-repos \
  --capabilities CAPABILITY_IAM \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    HttpApiName=sports-models-http-api-v2 \
    ApiFunctionName=sports-models-api-v2 \
    TrainingFunctionName=sports-models-training-v2 \
    Localhost=False \
    EnvironmentName=PROD \
    AdminApiKey="$ADMIN_API_KEY" \
    FrontEndApiKey="$FRONT_END_API_KEY" \
    ReadApiKey="$READ_API_KEY" \
    NbaApiKey="$NBA_API_KEY" \
    AwsApiKey="$AWS_API_KEY" \
    CfbdApiKey="$CFBD_API_KEY" \
    SupabaseDbUrl="$SUPABASE_DB_URL" \
    SupabaseSchema="$SUPABASE_SCHEMA"
```

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
