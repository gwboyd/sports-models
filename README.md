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

## Quickstart

This is the shortest path to run both the backend and frontend locally.

### 1. Backend setup

Create the backend env file if you have not already:

```shell
cp .env.example .env
```

Set these values in `.env`:
- `LOCALHOST=True`
- `SUPABASE_DB_URL`
- `SUPABASE_SCHEMA=sports_models`
- API key values such as `ADMIN_API_KEY`, `FRONT_END_API_KEY`, `READ_API_KEY`, `NBA_API_KEY`, `AWS_API_KEY`

`LOCALHOST=True` bypasses API auth checks for local development, but the values still need to be present so startup does not fail.

Start the backend:

```shell
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 3000 --reload
```

Verify it:

```shell
curl http://127.0.0.1:3000/health
```

### 2. Frontend setup

Install frontend dependencies:

```shell
cd frontend
npm install
```

Create `frontend/.env` with:

```shell
ENDPOINT=http://127.0.0.1:3000
AUTHORIZATION_TOKEN=
```

Start the frontend:

```shell
cd frontend
npm run dev
```

Open:
- Backend: `http://127.0.0.1:3000`
- Frontend: `http://127.0.0.1:5173`
- NFL page: `http://127.0.0.1:5173/models/nfl`
- NBA page: `http://127.0.0.1:5173/models/nba?bankroll=500`
- Info page: `http://127.0.0.1:5173/models/info`

### 3. Common local issues

- If the frontend shows a stale runtime error after a fix, do a hard refresh once.
- If backend requests fail, confirm `SUPABASE_DB_URL` points at the pooled Supabase Postgres connection string.
- If port `3000` or `5173` is already in use, stop the existing process or choose a different port.
- When deployed on Vercel, the Remix route responses use `Cache-Control` headers so Vercel can cache the NFL/NBA pages for 5 minutes and the info page for 1 hour.

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
