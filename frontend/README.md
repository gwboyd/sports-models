# Frontend

Next.js frontend for viewing NFL, CFB, and NBA model outputs.

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
- `http://127.0.0.1:5173/models/nfl/results`
- `http://127.0.0.1:5173/models/nfl/how-it-works`
- `http://127.0.0.1:5173/models/nfl/insights`
- `http://127.0.0.1:5173/models/cfb`
- `http://127.0.0.1:5173/models/cfb/results`
- `http://127.0.0.1:5173/models/nba?bankroll=500`

`/models/info` is retained as a redirect to `/models/nfl/how-it-works`.

## Football UI

NFL and CFB share a mobile-first expected-points dashboard. Each slate is presented by game, with favorites first,
separate spread/total lock cards second, and the full chronological game list last. Desktop uses a semantic game
table; mobile uses expandable game cards. CFB games appear once and can be filtered by either team's conference.

Favorites are device-local and stored under `sports-models:favorites:v1`, separated by league. Search covers the
current league slate and matches team names, abbreviations, aliases, and CFB conferences.

Team identity metadata lives in `app/lib/team-data.ts`. Approved assets should be added as SVG or transparent PNG
files under `public/teams/<league>/` and enabled in the manifest. Until an asset is approved, the shared team component
renders a monogram fallback without making an external image request.

The results routes use the existing pick-results responses. They calculate season-specific summaries from graded game
rows, keep the selected season in `?season=`, and show a designed empty state when results are not available. The NFL
visitor explainer is maintained separately from developer documentation in `content/nfl-how-it-works.md`.

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
NFL, CFB, and NBA pages render at request time while their backend data is cached for
five minutes. This keeps deployments independent of backend availability.

The CFB games page fetches the current slate only. Its results route handles a missing result set as an empty state,
allowing the first season to launch before graded data exists.
