# sports-models

Backend services for sports prediction models, deployed on AWS Lambda with SAM and backed by Supabase Postgres.

## Overview

- FastAPI app served locally with `uvicorn` and in AWS through Lambda + API Gateway
- Supabase Postgres for persisted picks and results data
- Separate SAM-managed Lambdas for API traffic and long-running NFL training/update work

The operational database schema is `sports_models`.

## Setup

Use Python `3.12.x`.

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
- `SUPABASE_DB_URL`
- `SUPABASE_SCHEMA`

For local development, also set:

- `LOCALHOST=True`
- `SUPABASE_SCHEMA=sports_models`

Use the pooled Supabase Postgres connection string for `SUPABASE_DB_URL`.

## Database

Run:

```text
db/sql/001_create_sports_models_schema.sql
```

This creates the `sports_models` schema and the tables/views used by the backend:

- `nfl_expected_points_picks`
- `nfl_expected_points_latest_picks`
- `nfl_expected_points_results`
- `nfl_expected_points_pick_updates`
- `nfl_expected_points_latest_updates`
- `nba_first_basket_picks`

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
- one separate training Lambda for `POST /nfl-update-picks`
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
    SupabaseDbUrl="$SUPABASE_DB_URL" \
    SupabaseSchema="$SUPABASE_SCHEMA"
```

After deploy, verify the API:

```shell
aws cloudformation describe-stacks --region us-east-1 --stack-name sports-models-v2
curl https://your-api-id.execute-api.us-east-1.amazonaws.com/health
```
