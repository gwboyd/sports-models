# Project Overview
This repository hosts sports prediction systems with a FastAPI backend deployed to AWS Lambda via
Mangum/SAM, notebook-driven model workflows for NFL, CFB, and NBA, and a frontend for model consumption.
Operational persistence now lives in Supabase Postgres under the `sports_models` schema. Expected-points picks and
graded results retain the recipe version that generated the pick, including when an older locked pick is graded after a
newer deployment.

## Repository Structure
- `src/model_patterns/`: Reusable modeling patterns; `expected_points/` is currently shared by NFL and CFB.
- `src/sports/`: Sport- and model-specific implementations (NFL/NBA/CFB notebooks, handlers, utilities).
- `src/utils/db/`: Centralized Postgres data access for operational tables/views.
- `src/utils/`: Shared infrastructure utilities and Pydantic models.
- `frontend/`: Frontend app for serving model views.
- `tests/`: Python test modules for API/Lambda behavior.
- `events/`: Sample AWS SAM event payloads for local invocation.
- `python/`: Vendored Python packages for Lambda packaging/layer-style usage.
- `main.py`: FastAPI app entrypoint, auth/permissions, router mounting, and Mangum handler.
- `template.yaml`: AWS SAM template for the Lambda function and API event wiring.
- `Dockerfile`: Lambda-compatible container image build definition.
- `requirements.txt`: Python dependency list for backend/model runtime.
- `requirements-dev.txt`: Runtime dependencies plus pinned local test tooling.
- `db/sql/001_create_sports_models_schema.sql`: Supabase setup SQL for the `sports_models` schema.

## Build & Development Commands
```sh
brew install uv
uv venv
source .venv/bin/activate
brew install libomp
brew install python-certifi
uv pip install -r requirements-dev.txt
uv pip check
```

```sh
uvicorn main:app --host 0.0.0.0 --port 3000 --reload --log-level warning
```

```sh
sam local invoke "FastAPILambdaFunction"
sam local start-api
sam local invoke FastAPILambdaFunction -e events/get-health-event.json
```

```sh
pytest
```

```sh
cd frontend
npm install
npm run dev
npm run build
npm run typecheck
npm run codegen
```

## Code Style & Conventions
- Python: keep type hints on public surfaces, explicit logging, and small focused helpers.
- SQL access belongs in `src/utils/db/`, not inline in handlers.
- Notebook runtime cells should stay separated from notebook-only exploratory cells using explicit guards.
- TypeScript frontend uses strict mode.
- Python modules/files use `snake_case`.
- React component files use `PascalCase`.

## Architecture Notes
```mermaid
flowchart LR
  U[Client Browser] --> FE[Frontend]
  FE --> API[FastAPI on AWS Lambda]
  API --> NFL[NFL Router]
  API --> CFB[CFB Router]
  API --> NBA[NBA Router]
  NFL --> EP[Shared Expected Points Runtime]
  CFB --> EP
  EP --> NB[Papermill Notebook Execution]
  NB --> DB[(Supabase Postgres)]
  NFL --> DB
  CFB --> DB
  NBA --> DB
  NB --> TMP[/tmp outputs]
  API --> RESP[JSON Responses]
```

`main.py` creates the FastAPI app, validates API keys, and mounts sport/model routers. Operational reads and
writes go through `src/utils/db/sports_models_db.py`. NFL and CFB update endpoints execute their notebooks via the
shared runtime in `src/model_patterns/expected_points/runtime.py`. Shared tracking helpers validate picks, preserve
started games, calculate changes, and grade completed picks. `write_expected_points_run` persists the update record,
current picks, and newly graded results in one transaction so a failed run cannot partially commit. Initial Postgres
connection failures use bounded retries in `src/utils/postgres.py`.

NFL expected-points acquisition is isolated in `src/sports/football/nfl/expected_points/data_loader.py`: `nflreadpy`
loads source data as Polars, selects model fields, and converts to pandas at the notebook boundary with caching off.
CFB expected-points uses `src/sports/football/cfb/expected_points/cfbd_client.py` for direct authenticated REST
requests; do not add a conflicting Python SDK dependency.

Current expected-points schema model, mirrored for the `nfl` and `cfb` prefixes:
- `<league>_expected_points_picks`: latest pick per `(year_week, game_id)`
- `<league>_expected_points_latest_picks`: view over the most recent `year_week`
- `<league>_expected_points_pick_updates`: update-run history and JSON snapshots
- `<league>_expected_points_latest_updates`: latest update per `year_week`
- `<league>_expected_points_results`: graded outcomes
- `model_releases`: immutable NFL/CFB expected-points recipe releases and public/internal release notes

Expected-points recipe releases use independent `MAJOR.MINOR` versions for NFL and CFB. A major release changes
the model's methodology or expected behavior; a minor release is a smaller prediction-affecting change. Routine
retraining on newly available data is a run under the same version. Historical rows are attributed to the bootstrap
baseline `1.0`. Its archived release files describe the production recipes present when tracking began and explicitly
note that exact pre-versioning code revisions cannot be reconstructed. Picks, results, and update runs carry the model
version that produced them, so performance can be grouped by exact version or by major version. Locked picks preserve
their original version when later runs change the recipe.

CFB pick and result rows additionally retain nullable `home_conference` and `away_conference` values from the
CFBD schedule. The CFB frontend renders each game once and uses those fields for a conference filter; a
cross-conference game matches either team's conference.

The frontend is mobile-first and game-centered for NFL and CFB. Each current-slate page orders favorites, separate
spread/total lock cards, and the complete game list. Favorites are device-local under
`sports-models:favorites:v1`. Shared results pages calculate season and weekly summaries from existing graded game
responses, and the NFL section has nested public methodology and model-insight routes. Approved team logos are local
assets referenced by generated frontend manifests; missing assets use monogram fallbacks. Refresh CFB metadata and
assets with `make sync-cfb-teams YEAR=<season>` (using the root `CFBD_API_KEY`), NFL assets with
`make sync-nfl-teams`, or both with `make sync-football-teams YEAR=<season>`. The sync never writes API credentials to
frontend output; generated manifests live under `frontend/app/generated/` and cached assets under
`frontend/public/teams/<league>/`.
Frontend additions should preserve the compact design system: neutral surfaces, eight-pixel card radii, restrained
shadows, custom electric ink blue (`#0B5FCC`) interaction/lock accents, tabular numbers, and 44px minimum interactive targets.
Lock emphasis is a uniform medium-blue outline without an accent side border. Favorite game cards show that outline
and independent spread/total lock tags whenever either market qualifies.
Keep the football hero limited to the game-search field; favorite management belongs in the Favorites section. The
team manager uses a separately scrolling results region that resets to the top on each query so single matches remain
visible above the mobile keyboard. Shared mobile overlays must remain anchored to `window.visualViewport` and lock
document scrolling for the full open lifetime. Mobile lock rails must align their first card to the main content inset.
Football kickoff labels must render in the browser device timezone. NFL and CFB both expose `date_time` as
`America/New_York` wall time in `YYYY-MM-DD-HH:MM` format; convert raw CFBD UTC timestamps before persistence.
Model-update timestamps also render in the device timezone and include the applicable daylight/standard abbreviation.
Keep the NBA route functional while its global tab is temporarily hidden; NFL and CFB are the visible league tabs.
Results summary cards lead with spread/total locks and omit a standalone predicted-games tile. In favorites and mobile
game cards, emphasize the actionable spread/total picks and outline each locked market individually in the lock blue.
The public NFL methodology lives in `frontend/content/nfl-how-it-works.md` and intentionally preserves the detailed
production Info-page content. Its renderer must continue supporting headings through `h3`, lists, links, inline code,
and responsive images; keep backend and deployment operations in the model README.

Interactive expected-points notebook executions use `client_name="notebook"` and must remain read-only. Runtime/API
executions use a non-notebook client name and persist through the shared atomic writer. CFB eligibility is explicit:
at least one participant must be FBS and at least one real sportsbook must supply each of the current spread and total;
moneylines, opening lines, and multiple providers are optional. CFB score features use median sportsbook consensus
lines. Betting direction is chosen against consensus, then the execution line is shopped only within the largest
provider cluster spanning at most two points. Cluster ties prefer proximity to consensus and then provider
reliability. Unsupported exact quote ties use Bovada, William Hill, ESPN Bet, DraftKings, Caesars, then remaining
providers alphabetically. A market without two corroborating providers remains visible and trainable but cannot
become a lock. The execution line drives model edge, confidence, persistence, display, and grading. Synthetic CFBD
providers are not executable quotes, and CFBD provider ordering must never affect selection.
Neither football model may predict a game at or after kickoff, including a game with no pre-existing pick.

Expected-points training uses an outer chronological score holdout, a single inner chronological GridSearch split,
and one final refit on all completed games with the selected score parameters. Confidence classifiers train from the
outer out-of-time predictions, tune with their own single chronological split, and refit on that full holdout. Do not
replace this with random splitting or run a second score GridSearch during the final refit.
CFB confidence tuning scores candidate parameters with log loss and always includes the execution-line spread/total
edge. Opening-line, movement, market-depth, and no-vig moneyline feature groups remain out of production unless a
genuine out-of-time comparison improves pooled Brier score by at least one percent, does not worsen log loss, and
keeps every evaluated season within two percent of its baseline Brier score. Read-only CFB notebook runs execute and
report this secondary chronological classifier evaluation; production/API notebook runs skip it. No optional market
feature group is currently promoted.

League-level validation belongs in `src/sports/football/<league>/data_validation.py`; generic dataframe contracts
belong in `src/sports/data_validation.py`. Validate only feeds a model actually consumes. Production notebook runs use
`strict_data_validation=True`; warning mode is for feed investigation and must never weaken chronological/leakage
checks. Keep stable loading, validation, feature construction, and training mechanics in Python modules while using
notebooks for orchestration, exploration, charts, and persistence previews.
League-neutral football transforms belong in `src/sports/football/transforms/`; source-specific conversion and feed
shaping belong beside the NFL or CFB workflow. Shared NFL/CFB HTTP schemas belong in
`src/sports/football/expected_points_schemas.py`.

Prediction-affecting expected-points work must be recorded in the applicable
`src/sports/football/<league>/expected_points/UNRELEASED.md`. The file is intentionally empty when there is no
pending model release. A non-empty draft must be released as either a major or minor version by the next production
deployment; it cannot ship under the old version. Drafts use strict Markdown sections for public summary and
changes, with optional evaluation and internal notes. After a successful release, the deployment command archives
the exact draft under `releases/vN.N.md` and resets the working file. Documentation, frontend, API-output, database,
and infrastructure-only changes do not belong in the draft.

Production model deployments use the interactive `make sam-deploy` workflow only. It loads both drafts, prompts for
major/minor for every non-empty draft, keeps empty drafts at their latest version, prints both decisions, and requires
final confirmation. The working tree, including untracked files, must be completely clean so the deployed image
matches the recorded Git SHA. NFL and CFB share one training Lambda image, so both populated drafts must be released
in the same deployment. There is no GitHub Actions version gate, source fingerprint, or changed-path heuristic. After
SAM succeeds, the command uses bounded retries to verify that the active training Lambda contains both planned model
versions and the planned Git SHA; only then does it record release rows. A release is considered live only after its
first successful AWS pick update sets `first_pick_at`. The ignored `.aws-sam/model-release-plan.json` preserves exact
draft snapshots for recovery. `make sam-register-releases` re-verifies AWS before registering, while
`make sam-finalize-release-files` verifies the Supabase rows before archiving/resetting drafts and removing the plan.

## Testing Strategy
1. Run `pytest` for backend smoke coverage.
2. Run `sam local start-api` and hit endpoints using sample events under `events/`.
3. For frontend work, run `cd frontend && npm run typecheck && npm run build`.
4. For notebook-driven NFL or CFB changes, require a human-verified run of the update workflow before merging.
5. Verify persisted pick counts, update history, started-game preservation, graded results when applicable, and the
   corresponding read endpoints.

## Security & Compliance
- Never commit secrets; use `.env` locally and secret stores in deployment.
- `template.yaml` should stay parameterized; do not hardcode API keys or DB URLs.
- Backend auth remains API-key based via request headers.
- Supabase access is backend-only through the pooled Postgres connection string in `SUPABASE_DB_URL`.

## Agent Guardrails
- Documentation maintenance is mandatory and part of every change. Whenever behavior, architecture, APIs, schemas,
  environment variables, deployment, testing, or operational workflows change, update `AGENTS.md` and every
  applicable README in the same change by default. Do not wait for the user to request documentation updates.
  Review all affected documentation before finishing, even when the final determination is that no text change is
  necessary, and never knowingly leave documentation stale.
- Never modify large historical datasets under `src/sports/football/cfb/starter_pack_data/` unless explicitly requested.
- Treat notebooks in `src/**/notebook.ipynb` as high-risk for noisy diffs. Prefer Python module edits unless notebook changes are required.
- Do not reintroduce DynamoDB operational persistence or migration tooling unless explicitly requested.
- Keep SQL setup-oriented; assume `db/sql/001_create_sports_models_schema.sql` is the source of truth for Supabase setup.
- Keep model-release SQL idempotent and setup-oriented. Apply the reviewed release-tracking statements to an existing
  Supabase database before the first version-aware production deploy; do not add a migration framework. Preserve its
  single-transaction execution and the `model_version = '1.0'` defaults that keep legacy Lambda writes compatible
  during rollout.
- Do not alter vendored runtime dependencies in `python/` unless the task is specifically about packaging/layer updates.
- For deployment-impacting changes (`template.yaml`, `Dockerfile`, auth logic), require reviewer approval before merge.

## Extensibility Hooks
- Active backend env vars:
  - `LOCALHOST`
  - `ENVIRONMENT`
  - `PORT`
  - `ADMIN_API_KEY`, `FRONT_END_API_KEY`, `READ_API_KEY`, `NBA_API_KEY`, `AWS_API_KEY`
  - `CFBD_API_KEY`
  - `SUPABASE_DB_URL`, `SUPABASE_SCHEMA`
- Add new routers in `src/sports/<sport>/<league>/<model>/handler.py` and mount them in `main.py`.
- Add new DB access helpers in `src/utils/db/`.
- Add model-specific Pydantic schemas beside their API boundary; only truly application-wide schemas belong in a
  shared utility package.

## Further Reading
- [`README.md`](README.md)
- [`db/sql/001_create_sports_models_schema.sql`](db/sql/001_create_sports_models_schema.sql)
- [`template.yaml`](template.yaml)
- [`main.py`](main.py)
- [`src/sports/football/nfl/expected_points/README.md`](src/sports/football/nfl/expected_points/README.md)
- [`src/sports/football/cfb/expected_points/README.md`](src/sports/football/cfb/expected_points/README.md)
- [`frontend/README.md`](frontend/README.md)
