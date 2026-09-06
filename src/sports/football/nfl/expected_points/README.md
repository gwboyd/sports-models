# NFL Expected Points Model

The visitor-facing, copyedited explanation rendered by the frontend is maintained in
`frontend/content/nfl-how-it-works.md`. This README remains the source for developer and operational workflow details.

The model predicts **expected points scored** for each NFL team in a game, getting to a predicted score from which a spread and total can be derived. We then compare those spreads and totals to the Vegas prices to make the picks each week. 

Metrics derived from play by play data is used starting from 2010 to now. EPA (expected points added) Per Play and Success Rate are the features the model has found the most useful to predict outcomes, but other features like starting-quarterback NFL passer rating (retained under the historical `qbr` feature name), days of rest, and the Vegas odds themselves are used.

After picks are made, there is another model (classifier) that looks back on the historical picks the model has made against Vegas, analyzes patterns with which the mdoel has been succesful, and gives a percentage chance it belives the model has of being correct in it's pick. That score (along with a couple other heuristics) is how we decide what "plays" to make each week.

## Picks Update Cadence

A small EventBridge-driven planner derives updates from the nflverse schedule rather than fixed NFL weekdays. It
opens a week only after the prior week is final and the 5:00 AM Eastern rollover check has passed. A first week of a
new season switches to active cadence when it enters the four-day horizon.

An eligible NFL week trains daily at 5:00 AM Eastern through its final game date. Every date containing games receives
up to three additional updates: one hour before the first early game, first afternoon game, and first night game.
Early means before 2:00 PM Eastern, afternoon is 2:00-6:59 PM, and night begins at 7:00 PM. Empty buckets are skipped,
and the same rule covers Thursday, Monday, international, Saturday, holiday, and playoff schedules. Multiple games in
one bucket share the update before its earliest kickoff; a Saturday/Sunday slate can therefore have six game-window
updates in addition to the daily 5:00 AM runs.

The planner persists the week in `scheduled_model_updates` and creates or updates stable one-time EventBridge
schedules. Each schedule invokes training directly through IAM, and kickoff changes overwrite its future trigger
rather than creating a duplicate. The serialized training Lambda atomically claims the plan, then links the plan's
generic `(model_key, update_id)` fields to the persisted NFL update row in the same transaction as the picks. No API
Gateway request or API key is involved. With no published future game, weekly schedule-feed checks create no training
run. Once next-season games are published outside the four-day horizon, one weekly offseason training run is created;
inside four days, active cadence replaces it. Final-season picks are graded by that next successful notebook run, not
by the coordinator.

An existing prediction is locked 30 minutes before kickoff and preserved by later update runs. A game is removed from
the live prediction slate entirely once kickoff is reached.

Completed games are graded during later update runs when final scores become available. Graded outcomes are written
with the current picks and update history through the shared expected-points persistence workflow.

## Data Loading

The notebook receives its data from `data_loader.py`, which uses `nflreadpy` for play-by-play, weekly player stats,
schedules, and team metadata. The loader selects only model-consumed fields while data is still Polars, disables the
library cache, then converts the selected frames to pandas for the existing feature pipeline. It preserves the former
float32 PBP behavior, renames player-stat `team` and passing-interception fields to the notebook's legacy names, and
normalizes Oakland, San Diego, and St. Louis abbreviations. During week 1 only, an unavailable current-season
play-by-play or player-stat release is logged and skipped; missing historical data and later-season failures remain
errors. Strict input contracts additionally enforce nonempty loaded seasons, exact season coverage, unique player-week
and schedule keys, and complete team plotting metadata. Pass `strict=False` only to warn while investigating feed
drift; production notebook runs use strict mode.

Kickoffs are stored as `America/New_York` wall time for both football leagues. NFL schedule fields already use that
source timezone and are validated during frame preparation; comparisons with the current instant are timezone-aware.
Games at or after kickoff never enter the live prediction frame.

## Temporal Training and Evaluation

Completed games are sorted by kickoff. The latest 20% form an outer score-model holdout, so every confidence-training
prediction comes from games later than the score model's training data. Score-model GridSearch uses one chronological
validation split inside the earlier 80%, and each confidence classifier uses one chronological validation split
inside the outer holdout. The chosen score parameters are then refit once on all completed games for production;
GridSearch is not repeated. This keeps update-time cost modest while removing random-split leakage, but it is not a
full out-of-fold season backtest.

The notebook remains the interactive analysis surface while `recipe.py` supplies the reproducible configuration used
by historical runs. After training, `health_metrics` displays score-error, confidence, calibration, and weekly lock
metrics, and `health_results` contains the untouched confidence-evaluation rows. `inspect_game(game_id)` returns the
complete model row, exact home/away score representations, confidence inputs, and final pick;
`inspect_team_history(team, before_game_id=...)` returns earlier team-game rows for feature investigation.

Set `write_backtest_frame=True` and run through the dedicated frame-save cell to write `df` without running the normal
model train. Training, game inspection, and the optional historical comparison are separate notebook stages. Set
`run_historical_backtest=True` only for an interactive `quick`, `standard`, or `full` comparison; Papermill/API runs
reject it. The saved frame also supports
`make backtest-expected-points LEAGUE=nfl PROFILE=standard BASELINE=deployed`. Compatible local cutoff results under
`.backtests/expected_points/` are appendable and reusable; recipe/configuration changes or corrected prior inputs
invalidate affected fits. Working-tree fits are fresh by default; set notebook parameter
`backtest_cache_working_tree=True` or Make variable `CACHE_WORKING_TREE=1` only when resumable candidate caching is
desired. Historical line and roster values are the best currently available feed values, not exact
intraday snapshots. See the root [`backtesting guide`](../../../../../docs/expected-points-backtesting.md) for
baseline references, expense controls, artifacts, cache behavior, and metric interpretation.

`NFLExpectedPointsRecipe` owns final schedule/score/feature-frame joins, column normalization, kickoff conversion,
line orientation, validation, feature selection, tuning grids, and lock thresholds. The notebook calls that assembly
while retaining `scores`, `schedule_scores`, and `df` for inspection. When refs need different prepared feature
columns, provide `BASELINE_FRAME` and `CANDIDATE_FRAME`; every cutoff is still refit and both frames must describe the
same game/outcome/market universe.

## Operational Update Workflow

The NFL notebook shares its tracking, grading, reporting, notebook execution, and database persistence behavior with
the CFB expected-points model through `src/model_patterns/expected_points/`.
The shared tracking configuration supports league-specific pick metadata so CFB can retain home and away conference
values without changing the NFL pick schema.
League-neutral lagged-metric calculations live under `src/sports/football/transforms/`; nflverse-specific kickoff
formatting and NFL feature shaping remain in this workflow.

Each operational update:

1. Produces and validates the current predictions.
2. Loads existing NFL picks and results from Supabase.
3. Preserves saved predictions for games inside the kickoff lock window.
4. Records changed picks and plays in the update-history snapshot.
5. Grades previously saved picks whose games have completed.
6. Writes the update record, current picks, and newly graded results in one transaction.

Only the deployed AWS training Lambda writes automatically. Interactive runs default to `client_name="notebook"`
and `allow_non_aws_write=False`, so they remain read-only. To perform an intentional notebook write, set
`allow_non_aws_write=True`; the notebook resolves the latest registered NFL version and requires the exact
`WRITE NFL <VERSION>` confirmation before using the shared atomic writer. Local API and SAM-local update requests return `403` and cannot create production schedules. Explicit season/week
and non-AWS write options remain available to the internal notebook runner, not the HTTP endpoint. Changes to this workflow require a
human-verified update run before merging, including checks for pick counts, update history, locked-game preservation,
graded results when applicable, and the read endpoints.

## On-demand refresh

`POST /nfl-update-picks` on the deployed API accepts no season/week body. Use admin `Authorization`, `client-name`,
and an optional `Idempotency-Key`; the interactive client name `notebook` is reserved and rejected here. It selects the current eligible model slate with the same calendar/rollover policy
as the coordinator and creates one EventBridge schedule approximately one minute later. A confirmed submission
returns `202` with the resolved season/week, run key, and status URL; no eligible upcoming slate returns
`200` / `not_scheduled`. Old explicit-week bodies return `422`.

`GET /model-update-jobs/{run_key}` is an admin-only status read. Reusing the same POST key returns the original job;
it never moves the target week/time or retrains a completed job. A new key requests a fresh run. The response echoes
the effective key, including on scheduling errors. Manual jobs share `scheduled_model_updates` and the existing
versioned atomic writer, retaining the requesting client and the deployed version/SHA at execution time. Calendar
reconciliation excludes these manual jobs. A delayed job is cancelled if a newer model week has already published.

A failed attempt may still be retried by AWS. Stale jobs are marked `outcome_unconfirmed` in the response rather than
claiming success or terminal failure; inspect existing logs and failure queues before requesting another run. Apply
the additive run-table `trigger_source`/`client_name` SQL before deployment; the deployment command checks those
columns before building or changing AWS. See the root README's on-demand update
section for request examples, idempotency recovery, timing, status semantics, and rollout checks.

## Model Release Queue

Prediction-affecting NFL changes are recorded in `UNRELEASED.md`. Keep that file empty when the deployment contains no
NFL recipe change; do not add frontend, documentation, API-output, infrastructure, or database-only work. A populated
draft must contain `#` title, `## Public Summary`, and `## Changes` sections, with optional `## Evaluation` and
`## Internal Notes` sections.

Production deployment is performed through `make sam-deploy`. The command prompts for `major` or `minor` whenever
this queue is non-empty, keeps the current version when it is empty, prints the NFL and CFB decisions together, and
requires confirmation, a completely clean Git tree, and the checked-out `main` branch. Both models share one training Lambda image, so a populated
NFL queue ships in the same AWS deployment as CFB. After SAM succeeds, the command verifies the active Lambda's model
versions and Git SHA before registering the draft in the Supabase `model_releases` table and archiving it as
`releases/vN.N.md`; the working queue is then reset. When a kept bootstrap release still has no registry SHA, this
verified registration step initializes it exactly once without moving an existing SHA. A release is considered live
only after its first successful AWS pick update records `first_pick_at`.

## Features

The model is primarily powered by EPA per play and Success Rate (broken up by offense/defense and pass/rush). Both of these are efficiency metrics that are impressively predictive.

**EPA (Expected Points Added):** Measures how many expected points a play added to a total. The sum of all expected points added for plays in a drive should be 7, with the more impactful plays relative to expectations carrying more value.

**Success Rate:** "Success" is a tightly defined binary metric that determines if a play is successful or not dependent on the situation the offense is in. I used a custom success rate formula for this model that is as follows:
- On 1st down if yards gained was more than *0.4 * yards to go* (More than 4 yards on a first and 10)
- On 2nd down if yards gained was more than *0.6 * yards to go* (More than 6 yards on a second and 10)
- On 3rd and 4th down if the first down marker was reached

In the **charts** section, there should be some graphics that show Offensive and Defensive EPA metrics for teams as the model currently sees them. These can be helpful if you are trying to peel back the curtain and understand why the model made a certain pick.

### Feature Importance

Sometimes in machine learning, we will throw a bunch of features at the model and it will choose a "set of weights" for the features that result in the smallest amount of error. This specific model uses a gradient-boosted framework, so that's not exactly how it works, but we can still see overal feature importances which can give us a good idea of what the model finds predictive.

![Model feature importances](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/feature_importance.png)

These importance can technically change as more data gets introduced, but the model has seen enough games to where the feature importance remains fairly steady. It's also interesting to me that the model finds the efficiency metrics more predictive than anything origionating from Vegas data.

The **pred_team** feature is indicitive how important the model thinks home field advantage is.

### Moving Averages for Metrics

We calculate the advanced metrics using play by play data, but end up aggregating up to a game. From there, we take an exponentially weighted moving average to smoothe other some of the variance. Finally, we make the window over which weight games dynamic. Typically, it averages over the whole season, but if it is earlier than week 10, it will use the last 10 games, going into the previous year. 

Below is an example of the various calculations for the Dallas Cowboys. It should be fairly obvious why opt to smoothe out the metrics so much.

![Dynamic Window Example](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/dynamic_window_example.png)

## Determining what Picks to Play

As I mentioned, there is a 2nd model that observes what types of games the model is good at picking, and what types of games with which it has struggled to beat Vegas. There is a separate model for spreads and totals.

The training dataset for these classifiers is the later, chronological test set of the 80/20 split. We cannot use predictions of games the score model has been trained on, since the model would have seen them before. The classifier therefore learns only from genuine out-of-time score predictions, while its own parameter search also keeps validation games after its training games.

**A common question I get asked is "Why is the model not confident in the pick even though the predicted spread is so far off the Vegas spread?"** The answer lies in that the "confidence score" comes from this objective 3rd party model, and in a case of high diofference of spreads/totals and low confience, is saying it has seen similar scenarios before where the model has lost and it is therefore not that confident in the pick.


## Misc

### Power Rankings

The power rankings seen above in the chart are created by taking all of the metrics (so not inclusing odds, enviroment, rest, etc) for teams, training a classification model that predicts winning, and them creating simulations to where each team plays every other team home and away. The model outputs a win probability for esach game, so the teams "win percentage" is just and average of the win probabilities for all 62 (31 * 2) games a team would play.

The game simulations mimick each team's form for the current week (the next week if a team is on a bye), so it would be as if they all played eachother "today."
The power-ranking classifier tunes against one chronological train-before-validation split; it is supplemental chart
logic and does not feed the expected-points picks or confidence classifiers.

### Ideas for the future

- Add in more player specific data besides just quarterbacks to better account for injuries/trades
- Adjust efficiency metrics for difficulty of opponent
- Include explosiveness data 
- Add some 3rd down specifc metrics data
- Include some position group specific data to better catch matchup advantages
- Revamp how metrics are handled for rookies (only player specific metric right now is QBR)
- Get to a place where I can remove any information that relies on Vegas odds.. it helps but sorta feels like cheating
- If I do keep Vegas lines, find a way to get opening odds or where the public is so I better arbitrage the "vibes vs metrics" dynamic
