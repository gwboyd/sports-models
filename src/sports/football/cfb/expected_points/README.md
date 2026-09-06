# CFB Expected Points Model

The CFB expected-points notebook uses the shared expected-points training, tracking, grading, and Supabase
persistence workflow. CFB-specific acquisition and feature preparation remain in this directory. Production notebook
runs use strict validation, chronological training splits, and the direct CFBD client described in the repository
README and `AGENTS.md`.

The offense and defense explosiveness features retain their existing names, but their values are now adjusted for
opponent quality before the existing dynamic moving average is calculated. The shared transform estimates
ridge-regularized offense and defense effects from games strictly before each target kickoff, uses FBS/FCS group
effects, and supports either partial pooling of individual FCS teams or one pooled FCS entity. Ridge strength,
cross-season carryover, and the FCS policy are explicit backtest levers.
Ratings default to the opening kickoff of the game week, excluding same-week outcomes; exact per-kickoff snapshots
remain available when evaluating that additional cost.
Set notebook/Papermill `opponent_adjustment_config` to a mapping such as
`{"ridge_alpha": 10.0, "season_carryover": 0.25, "fcs_policy": "pooled"}` to evaluate an alternative.

The notebook remains the interactive analysis surface. Its reproducible configuration lives in `recipe.py`, while
`schedule`, `market_lines`, intermediate feature frames, the complete `df`, `ep_config`, `results`, and `plays` remain
available for direct slicing. After training, `health_metrics` displays chronology-safe score, probability,
calibration, and weekly lock results. `inspect_game(game_id)` returns the market/model row, exact score and confidence
features, selected execution quote, prediction, and lock decision; `inspect_team_history` shows earlier team-game rows.

Interactive users may set `write_backtest_frame=True` and run through the dedicated frame-save cell without running
the normal model train. Training, game inspection, and comparison are separate stages. Set
`run_historical_backtest=True` only to launch `quick`, `standard`, or `full`; Papermill/API runs reject the expensive
option. The notebook-saved frame also supports
`make backtest-expected-points LEAGUE=cfb PROFILE=standard BASELINE=deployed`. Persistent cutoff artifacts live under
ignored `.backtests/expected_points/` and are invalidated by recipe/configuration or relevant historical-input
changes. Working-tree fits are fresh by default; set notebook parameter `backtest_cache_working_tree=True` or Make
variable `CACHE_WORKING_TREE=1` only when resumable candidate caching is desired. CFBD has no point-in-time provider
snapshots, so reports label this as weekly historical approximation. See
the root [`backtesting guide`](../../../../../docs/expected-points-backtesting.md) for baseline references, expense
controls, artifacts, cache behavior, and metric interpretation.

`CFBExpectedPointsRecipe` owns final schedule/pregame-feature/market joins, eligibility filtering, kickoff and column
normalization, validation, feature selection, tuning grids, confidence inputs, betting transform, and lock thresholds.
The notebook keeps `joined_features` and `df` visible. Versions needing different prepared feature columns can use
`BASELINE_FRAME` and `CANDIDATE_FRAME`; cutoff fitting remains fresh and both evaluation universes must match.

## Picks Update Cadence

A small EventBridge-driven planner derives updates from the CFBD regular-season schedule rather than fixed CFB
weekdays. It keeps the current week active until its applicable games are final and the 5:00 AM Eastern rollover check
has passed on the following morning. A new season switches to active cadence inside the four-day horizon.

An eligible CFB week trains daily at 5:00 AM Eastern through its final game date. Every date containing games receives
up to three additional updates: one hour before the first early game, first afternoon game, and first night game.
Early means before 2:00 PM Eastern, afternoon is 2:00-6:59 PM, and night begins at 7:00 PM. Empty buckets are skipped,
and the same rule covers every weekday without maintaining separate cron rules. Multiple games within one window
share the update before its earliest kickoff.

The planner stores the week in `scheduled_model_updates` and creates or updates stable one-time EventBridge schedules.
Those schedules invoke the training Lambda directly through IAM. Kickoff changes overwrite the future trigger instead
of adding one, and the training Lambda atomically claims its deterministic run identity before Papermill starts. Its
database transaction links the plan's generic `(model_key, update_id)` fields to the persisted CFB update row. It never
uses API Gateway or an API key. With no published future game, weekly schedule-feed checks create no training run.
Once next-season games are published outside the four-day horizon, one weekly offseason training run is created;
inside four days, active cadence replaces it. Final-season picks are graded by that next successful notebook run, not
by the coordinator.

## Model Release Queue

Prediction-affecting CFB changes are recorded in `UNRELEASED.md`. Keep that file empty when the deployment contains no
CFB recipe change; do not add frontend, documentation, API-output, infrastructure, or database-only work. A populated
draft must contain `#` title, `## Public Summary`, and `## Changes` sections, with optional `## Evaluation` and
`## Internal Notes` sections.

Production deployment is performed through `make sam-deploy`. The command prompts for `major` or `minor` whenever
this queue is non-empty, keeps the current version when it is empty, prints the NFL and CFB decisions together, and
requires confirmation, a completely clean Git tree, and the checked-out `main` branch. Both models share one training Lambda image, so a populated
CFB queue ships in the same AWS deployment as NFL. After SAM succeeds, the command verifies the active Lambda's model
versions and Git SHA before registering the draft in the Supabase `model_releases` table and archiving it as
`releases/vN.N.md`; the working queue is then reset. When a kept bootstrap release still has no registry SHA, this
verified registration step initializes it exactly once without moving an existing SHA. A release is considered live
only after its first successful AWS pick update records `first_pick_at`.

Only the deployed AWS training Lambda writes automatically. Interactive runs default to `client_name="notebook"`
and `allow_non_aws_write=False`, so they remain read-only. To perform an intentional notebook write, set
`allow_non_aws_write=True`; the notebook resolves the latest registered CFB version and requires the exact
`WRITE CFB <VERSION>` confirmation before using the shared atomic writer. Local API and `sam local` calls are also
read-only unless their request explicitly enables `allow_non_aws_write`.
