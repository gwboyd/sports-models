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
The NBA route remains directly accessible, but its global navigation tab is temporarily hidden. The visible product
navigation is limited to NFL and CFB.

## Football UI

NFL and CFB share a mobile-first expected-points dashboard. Each slate is presented by game, with favorites first,
separate spread/total lock cards second, and the full chronological game list last. Desktop uses a semantic game
table; mobile uses expandable game cards. CFB games appear once and can be filtered by either team's conference.

Kickoff and model-update times are displayed in the browser device's timezone, including the applicable seasonal
timezone abbreviation on update timestamps. Until the backend emits one timezone-aware ISO 8601
contract for both leagues, the frontend treats CFB `date_time` values as UTC and NFL values as
`America/New_York` wall time, including daylight-saving transitions. Server rendering uses each feed's source
timezone, then switches to the device timezone after hydration without creating a hydration mismatch.

The shared visual system is intentionally dense and restrained: neutral canvas and surfaces, eight-pixel card
corners, minimal shadows, and a custom electric ink blue (`#0B5FCC`) for interactions and lock emphasis. Lock and qualifying
favorite cards use a medium-blue outline and compact labels instead of a separate warning color. Keep 44px touch
targets even when reducing surrounding padding.
Favorite cards lead with larger spread and total pick blocks above the model score; each qualifying market receives
its own blue outline. The same market-level blue outline identifies locks in mobile all-game cards.

Favorites are device-local and stored under `sports-models:favorites:v1`, separated by league. Search covers the
current league slate and matches team names, abbreviations, aliases, and CFB conferences.
Favorite cards with a qualifying spread and/or total use the same light-blue outline as lock cards and display a
separate compact tag for each qualifying market.
The slate header exposes only the game-search field; favorite-team management opens from the Favorites section's
`Edit teams` action. The manager keeps its search field fixed above an independently scrolling result list and resets
that list to the top whenever the query changes. Mobile sheets lock the document for their full open lifetime and
follow the browser's visual viewport, keeping filtered results above the iOS software keyboard.

Team metadata is generated into `app/generated/cfb-teams.json` and `app/generated/nfl-teams.json`; the shared resolver
in `app/lib/team-data.ts` consumes those files at build time. Logo images are cached under
`public/teams/<league>/`, so browsers never need to call CFBD, nflverse, or ESPN. Unmatched feed names and unavailable
assets retain the monogram fallback.

Refresh the 2026 CFB catalog from the repository root with:

```shell
make sync-cfb-teams YEAR=2026
```

The target loads `CFBD_API_KEY` from the root `.env`, calls the FBS teams endpoint, selects the standard 128px logo
when available, and atomically rewrites the deterministic manifest after caching the images. The bearer token is not
written into generated output. NFL uses the same logo source already consumed by the model notebook
(`nfl_data_py.import_team_desc()` / nflverse):

```shell
make sync-nfl-teams
make sync-football-teams YEAR=2026
```

Commit changed manifests and logo assets after reviewing them. Re-running a sync is safe; unchanged manifests are not
rewritten, transient logo failures retain an existing cached file, and no account or runtime API credential is exposed
to the frontend.

The results routes use the existing pick-results responses. They calculate season-specific summaries from graded game
rows, keep the selected season in `?season=`, and show a designed empty state when results are not available. Summary
cards lead with spread and total locks, followed by all spread and total picks; there is no separate predicted-games
tile because every metric already includes its graded-pick count. The NFL
visitor explainer is maintained separately from developer documentation in `content/nfl-how-it-works.md`.
That document carries forward the public methodology previously rendered by the production Info page. Its renderer
supports nested headings, lists, links, inline code, and responsive Markdown images; operational database and
deployment instructions remain in the model README instead.
Model Insights is limited to the live power-ranking and offensive/defensive EPA views. Feature importance and the
dynamic moving-average explanation remain in How It Works.

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
