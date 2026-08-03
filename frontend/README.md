# Frontend

Next.js frontend for viewing NFL and NBA model outputs.

## Setup

Install dependencies:

```shell
npm install
```

Create `frontend/.env` (or `.env.local`):

```shell
ENDPOINT=http://127.0.0.1:3000
AUTHORIZATION_TOKEN=
```

- `ENDPOINT` is the backend base URL Server Components call
- `AUTHORIZATION_TOKEN` is sent as the backend `Authorization` header when present
- Both values are server-only. Do not rename either with a `NEXT_PUBLIC_` prefix.

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

The frontend dev server runs at `http://127.0.0.1:5173`, leaving port `3000` available for FastAPI.

Next.js 16 requires Node.js `20.9` or newer.

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

Lint and unit tests:

```shell
npm run lint
npm run test
```

Browser tests use a local mock API:

```shell
npx playwright install
npm run test:e2e
```

## Vercel

Keep `frontend` as the Vercel project root and allow Vercel to detect Next.js. Set
`ENDPOINT` and `AUTHORIZATION_TOKEN` in both Preview and Production environments.
NFL and NBA pages render at request time while their backend data is cached for
five minutes. This keeps deployments independent of backend availability.
