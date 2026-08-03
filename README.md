# sports-models

Backend services for sports prediction models, deployed on AWS Lambda with SAM and backed by Supabase Postgres.

## Overview

- FastAPI app served locally with `uvicorn` and in AWS through Lambda + API Gateway
- Supabase Postgres for persisted picks and results data
- Separate SAM-managed Lambdas for API traffic and long-running NFL/CFB training updates

The operational database schema is `sports_models`.

## Setup

Use Python `3.11.x` to match the Lambda container runtime.

Create and activate a virtual environment:

```shell
python3 -m venv .venv
source .venv/bin/activate
```

Install system dependencies:

```shell
brew install libomp
brew install python-certifi
```

Install Python dependencies:

```shell
pip3 install -r requirements.txt
```

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
frame.

The mobile-first NFL and CFB frontend presents favorites, separate spread/total lock cards, and a single game-centered
slate. CFB games appear once and can be filtered by either team's conference. Shared results routes derive season and
weekly summaries from the existing graded-game responses; an unavailable CFB result set renders an empty state until
the first games are graded. NFL methodology and live model graphics are available on separate nested routes.

Football team metadata and local logo assets can be refreshed without a frontend API change. `make sync-cfb-teams
YEAR=2026` loads `CFBD_API_KEY` from the root `.env`, writes a deterministic CFB manifest for the selected FBS season,
and caches its logos under `frontend/public/teams/cfb/`. `make sync-nfl-teams` mirrors the nflverse/ESPN logo metadata
already used by the NFL notebook. `make sync-football-teams YEAR=2026` refreshes both catalogs.

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
