# Frontend

Remix frontend for viewing NFL and NBA model outputs.

## Setup

Install dependencies:

```shell
npm install
```

Create `frontend/.env`:

```shell
ENDPOINT=http://127.0.0.1:3000
AUTHORIZATION_TOKEN=
```

- `ENDPOINT` is the backend base URL the Remix loaders call
- `AUTHORIZATION_TOKEN` is sent as the backend `Authorization` header when present

## Local Testing

Run the frontend using the values in `frontend/.env`:

```shell
make frontend
```

That is the right choice when `frontend/.env` points at a deployed API.

Override the backend target for direct local backend testing:

```shell
make frontend-local
```

That uses `http://127.0.0.1:3000`.

Override the backend target for local SAM testing:

```shell
make frontend-sam
```

That uses `http://127.0.0.1:3001`.

If you prefer not to use `make`:

```shell
npm run dev
```

The frontend dev server runs at `http://127.0.0.1:5173`.

Useful routes:

- `http://127.0.0.1:5173/models/nfl`
- `http://127.0.0.1:5173/models/nba?bankroll=500`
- `http://127.0.0.1:5173/models/info`

## Checks

Typecheck:

```shell
npm run typecheck
```

Production build:

```shell
npm run build
```
