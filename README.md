# sports-models

Backend services for sports prediction models, deployed on AWS API Gateway + Lambda and backed by Supabase Postgres.

## Architecture

- AWS API Gateway
- AWS Lambda running FastAPI
- Supabase Postgres for persisted picks/results data

The database schema for operational data is `sports_models` inside a Supabase project.

Database access is centralized in `src/utils/db/` so handlers and notebooks do not embed SQL directly.

## Getting Started

Use Python `3.12.x`.

### Create and activate a virtual environment

```shell
python3 -m venv .venv
source .venv/bin/activate
```

### Install system dependencies

```shell
brew install libomp
brew install python-certifi
```

### Install Python dependencies

```shell
pip3 install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the required values:

```shell
cp .env.example .env
```

Required values:
- `ADMIN_API_KEY`
- `FRONT_END_API_KEY`
- `READ_API_KEY`
- `NBA_API_KEY`
- `AWS_API_KEY`
- `SUPABASE_DB_URL`
- `SUPABASE_SCHEMA`

Use the pooled Supabase Postgres connection string for `SUPABASE_DB_URL`.

## Supabase Setup

Run the SQL in:

```text
db/sql/001_create_sports_models_schema.sql
```

This creates the `sports_models` schema and the operational tables/views used by the app:
- `nfl_expected_points_picks`
- `nfl_expected_points_latest_picks`
- `nfl_expected_points_results`
- `nfl_expected_points_pick_updates`
- `nfl_expected_points_latest_updates`
- `nba_first_basket_picks`

Current NFL data model:
- `nfl_expected_points_picks`: latest pick per `(year_week, game_id)`
- `nfl_expected_points_latest_picks`: view of picks for the most recent `year_week`
- `nfl_expected_points_pick_updates`: run history, diffs, and JSON snapshots for each update
- `nfl_expected_points_results`: graded outcomes for completed games

## Local Development

Run the API locally with:

```shell
uvicorn main:app --host 0.0.0.0 --port 3000 --reload
```

## AWS SAM Deployment

The SAM template expects secrets to be supplied as parameters rather than hardcoded in the template.

Example:

```shell
sam deploy \
  --parameter-overrides \
    AdminApiKey=... \
    FrontEndApiKey=... \
    ReadApiKey=... \
    NbaApiKey=... \
    AwsApiKey=... \
    SupabaseDbUrl=... \
    SupabaseSchema=sports_models
```
