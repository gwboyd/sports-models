# CFB Expected Points Model

The CFB expected-points notebook uses the shared expected-points training, tracking, grading, and Supabase
persistence workflow. CFB-specific acquisition and feature preparation remain in this directory. Production notebook
runs use strict validation, chronological training splits, and the direct CFBD client described in the repository
README and `AGENTS.md`.

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
