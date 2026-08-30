# CFB Expected Points Model

The CFB expected-points notebook uses the shared expected-points training, tracking, grading, and Supabase
persistence workflow. CFB-specific acquisition and feature preparation remain in this directory. Production notebook
runs use strict validation, chronological training splits, and the direct CFBD client described in the repository
README and `AGENTS.md`.

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
`releases/vN.N.md`; the working queue is then reset. A release is considered live only after its first successful AWS
pick update records `first_pick_at`.

Only the deployed AWS training Lambda writes automatically. Interactive runs default to `client_name="notebook"`
and `allow_non_aws_write=False`, so they remain read-only. To perform an intentional notebook write, set
`allow_non_aws_write=True`; the notebook resolves the latest registered CFB version and requires the exact
`WRITE CFB <VERSION>` confirmation before using the shared atomic writer. Local API and `sam local` calls are also
read-only unless their request explicitly enables `allow_non_aws_write`.
