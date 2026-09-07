# NFL Expected Points Model

The visitor-facing, copyedited explanation rendered by the frontend is maintained in
`frontend/content/nfl-how-it-works.md`. This README remains the source for developer and operational workflow details.

The model predicts **expected points scored** for each NFL team in a game, getting to a predicted score from which a spread and total can be derived. We then compare those spreads and totals to the Vegas prices to make the picks each week. 

Metrics derived from play by play data is used starting from 2010 to now. EPA (expected points added) Per Play and Success Rate are the features the model has found the most useful to predict outcomes, but other features like starting-quarterback NFL passer rating (retained under the historical `qbr` feature name), days of rest, and the Vegas odds themselves are used.

`build_pregame_quarterback_metrics` constructs quarterback history in kickoff order and keeps the existing
`ewma_qbr`/passer-rating feature names. Debut and unknown starters remain missing; never fill from a full-dataset
average of first starts. Score LightGBM handles missing inputs natively, and confidence imputation is fitted inside
each chronological training split. The September 2026 production audit removed the old future-dependent fallback;
the saved NFL 2.0 comparison predates that fix. A fresh performance comparison was explicitly waived by the user,
so do not attribute those measured gains to the final quarterback-history recipe.

The existing EPA and success-rate feature names now contain opponent-adjusted values. Before each game's kickoff,
the shared football transform fits ridge-regularized offense and defense effects only from earlier games, reassesses
each prior team performance against the defense or offense it faced, and then applies the existing moving-average
calculation. The adjustment therefore adds no future-game information and leaves downstream score and confidence
schemas unchanged. Its ridge penalty and cross-season carryover are recipe parameters for backtesting. Ratings use
the first kickoff of each game week by default, which excludes all same-week outcomes; the more expensive exact
kickoff mode remains available for evaluation.
In a notebook or Papermill run, set `opponent_adjustment_config` to partial overrides of the league defaults, such as
`{"ridge_alpha": 10.0, "season_carryover": 0.25}`; CFB additionally accepts
`{"fcs_policy": "pooled"}`.

The current NFL candidate uses half-strength correction (`adjustment_strength=0.5`), selected on 2023–2024
screening weeks and evaluated on the standard 2023–2025 comparison recorded in
`.backtests/expected_points/reports/expected-points-2.0-evaluation.md` (local, Git-ignored).
`adjustment_strength` scales the opponent correction from zero (unadjusted history) to one (full correction); the moving-average calculation is unchanged. `excluded_seasons` removes listed seasons from both rating estimation and team efficiency history. A zero carryover with no current-season observations uses a neutral correction instead of attempting a zero-weight fit. These controls do not remove score-training games.

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
validation split inside the earlier 80%. Version 2.1 probability heads fit separate absolute margin/total error
distributions on the outer holdout, discounted 80% toward an even chance; they do not run a classifier GridSearch. The chosen score parameters are then refit once on all completed games for production;
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
reject it. The standard CLI command automatically prepares its own frames:
`make backtest-expected-points LEAGUE=nfl`. Compatible local cutoff results under
`.backtests/expected_points/` are appendable and reusable; recipe/configuration changes or corrected prior inputs
invalidate affected fits. Working-tree fits are fresh by default; set notebook parameter
`backtest_cache_working_tree=True` or Make variable `CACHE_WORKING_TREE=1` only when resumable candidate caching is
desired. Historical line and roster values are the best currently available feed values, not exact
intraday snapshots. See the root [`backtesting guide`](../../../../../docs/expected-points-backtesting.md) for
baseline references, expense controls, artifacts, cache behavior, and metric interpretation.

`NFLExpectedPointsRecipe` owns final schedule/score/feature-frame joins, column normalization, kickoff conversion,
line orientation, validation, feature selection, tuning grids, and lock thresholds. The notebook calls that assembly
while retaining `scores`, `schedule_scores`, and `df` for inspection. Different feature columns are supported automatically. Optional
`BASELINE_FRAME`/`CANDIDATE_FRAME` overrides reuse explicit saved inputs; both frames must still describe the same
game/outcome/market universe.

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

`POST /nfl-update-picks` on the deployed API takes no body (or `{}`) and requires admin `Authorization` and
`client-name` headers. It selects the current eligible slate and schedules the existing trainer 60–120 seconds later.
A confirmed submission returns `202` with the selected season/week, run key, and status URL; no eligible slate returns
`200` / `not_scheduled`. Explicit-week bodies and the reserved client name `notebook` return `422`.

Supply an `Idempotency-Key` for safe retries. Reusing it retrieves the same job; a new key requests another run.
Poll admin-only `GET /model-update-jobs/{run_key}` for state, errors, and the completed update's version/SHA and counts.
Manual jobs retain the requesting client and use the trainer's version at execution time. Older protected picks keep
their original versions.

See [On-demand updates](../../../../../README.md#on-demand-updates) for request examples, retry/status semantics,
and the required database additions and deployment checks under [Scheduled Pick Updates](../../../../../README.md#scheduled-pick-updates).

## Model Release Queue

Keep the public [How It Works explanation](../../../../../frontend/content/nfl-how-it-works.md) current in
the same change as each model version's release notes. It describes the whole model (inputs, training, predictions, confidence, updates, and limitations), with ridge
adjustment as one part; `UNRELEASED.md` describes only user-visible differences. Review the other league's page
when shared behavior changes and coordinate frontend publication with the backend release. Keep large reports in
Git-ignored `.backtests/expected_points/reports/`, with detailed comparisons/experiments in their existing folders.
Those local reports are not shipped or pushed; do not force-add them.

Prediction-affecting NFL changes are recorded in `UNRELEASED.md`. Keep that file empty when the deployment contains no
NFL recipe change; do not add frontend, documentation, API-output, infrastructure, or database-only work. A populated
draft must contain a `#` title, `## Public Summary`, and `## Changes`. Describe shipped model differences in
plain language; omit code references, experiment settings, and evaluations. Keep evidence in separate reports, such
as `.backtests/expected_points/reports/expected-points-2.0-evaluation.md` (local, Git-ignored). Optional evaluation/internal
sections remain supported by the parser for compatibility, but are not part of new public drafts.

Before committing/merging a model release, run `make prepare-model-release` on the feature branch. It reads
registered versions, prompts for major/minor for each populated NFL/CFB draft, and confirms before copying the exact
notes to `releases/vN.N.md`, clearing the drafts, and updating root `model-versions.json`. Commit these files and the
updated whole-model How It Works pages with the implementation, then merge once. Preparation writes no Supabase rows
and does not deploy. If more prediction changes follow preparation, copy the unpublished release notes back into
UNRELEASED.md, update the complete draft and How It Works, and rerun prep. After confirmation it amends that same
unregistered version and clears the draft. Any nonblank draft blocks deployment. After registration, new prediction
changes require a new major/minor version. Do not manually clear drafts to bypass these checks.

Run `make sam-deploy` on clean `main` to deploy the committed versions. It rejects unprepared drafts, stale versions,
and changed registered notes, prints both versions, and requires confirmation. Both commands show changed source
and dependency inputs since the immutable registered SHA. Keeping the version despite those signals requires an
explicit prediction-neutral classification; they never choose a bump automatically. After SAM succeeds it verifies the
API, training, and coordinator Lambdas, registers/verifies releases in Supabase, and clears its ignored recovery plan.
It does not modify tracked files, so there is no second documentation merge. Existing canonical SHAs remain fixed;
a null bootstrap SHA is initialized only once. Supabase records actual deployment metadata and marks a release live
at its first successful AWS pick update (`first_pick_at`). The manifest and notes alone only indicate preparation.

For a failed build/deploy, rerun `make sam-deploy` at the same clean commit to resume its plan. After AWS succeeds,
Deployment rechecks the registry after confirmation and build to reject competing releases. Recovery validates both
league snapshots and their note hashes before verifying AWS or registering releases.
`make sam-register-releases` can recover registration without rebuilding or changing tracked files. See the root
README for recovery and preparation details. Prediction-neutral changes keep the manifest versions and empty drafts.


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

Version 2.1 separates score prediction from bet selection. Spread and total probabilities are `q=.5+.1*F(e)`, where
`e` is the absolute execution-line edge and `F` is the market-specific absolute score-error CDF on the chronological
score holdout. This assumes symmetric errors and discounts the inferred advantage 80%; it is not perfect calibration.
Locks require q>=.525 and positive expected units at assumed -110, including estimated pushes. The normal combined
weekly limit is three, with up to two spreads and one total; q>=.55 permits additional bets beyond those positive
caps. A zero market cap remains disabled. Started saved Locks consume capacity and retain their original version.
Volume is a target, not a forced minimum. See the public methodology for the complete explanation and limitations.

## Misc

### Power Rankings

The power rankings seen above in the chart are created by taking all of the metrics (so not inclusing odds, enviroment, rest, etc) for teams, training a classification model that predicts winning, and them creating simulations to where each team plays every other team home and away. The model outputs a win probability for esach game, so the teams "win percentage" is just and average of the win probabilities for all 62 (31 * 2) games a team would play.

The game simulations mimick each team's form for the current week (the next week if a team is on a bye), so it would be as if they all played eachother "today."
The power-ranking classifier tunes against one chronological train-before-validation split; it is supplemental chart
logic and does not feed the expected-points picks or confidence classifiers.

### Ideas for the future

- Add in more player specific data besides just quarterbacks to better account for injuries/trades
- Include explosiveness data 
- Add some 3rd down specifc metrics data
- Include some position group specific data to better catch matchup advantages
- Revamp how metrics are handled for rookies (only player specific metric right now is QBR)
- Get to a place where I can remove any information that relies on Vegas odds.. it helps but sorta feels like cheating
- If I do keep Vegas lines, find a way to get opening odds or where the public is so I better arbitrage the "vibes vs metrics" dynamic

### Evaluating local changes against deployed

**For any prediction-affecting change, just run `make backtest-expected-points LEAGUE=nfl` from the repository
root.** Adding, removing, renaming, or recalculating features uses the same command as changing training, estimators,
confidence inputs, or lock rules. It automatically prepares baseline and candidate features using each version's
code, validates their common historical evaluation universe, and runs the standard three-season weekly comparison.
It saves reports, prediction/lock changes, input bundles, notebooks, and logs in a new timestamped job directory.

No notebook run, `prepare-frame`, preflight, or quick run is required beforehand. `PROFILE=quick` is an optional
smaller run and repeats overlapping candidate fits if followed by standard with default caching. Optional controls
for dates/seasons, caching, saved frames, alternate baselines, and recovery are described in the
[backtesting guide](../../../../../docs/expected-points-backtesting.md#evaluate-working-tree-changes-against-deployed).
Keep code/settings fixed during evaluation and run the same command again after another model change. Opponent-adjusted
dynamic smoothing uses the target game's week for its span; older frames need regeneration after that correction.

The optional `smoothing_span` adjustment parameter overrides the base EWMA span (default: each metric's existing
10-game span); dynamic metrics still use the larger of that base span and the target week. Research CLI `--span`
sets the same lever. Changing it requires new candidate features and normal comparison evidence.

The shared standard comparison permits different training histories before the first evaluated season. It logs
those differences, keeps every side's training rows, and requires matching evaluation inputs and shared historical
outcomes/markets. Full-history identities remain attached to saved runs and caches. NFL training defaults are unchanged.

## Lightweight Lock replay

Use `make replay-expected-points-locks LEAGUE=nfl` to anchor to the deployed release's latest matching local
standard score run; `LOCK_SOURCE=version:2.0` or `artifact:<run>` pins another source. No expected-score refit is
needed. `LOCK_SCREEN_CADENCE=1 LOCK_MIN_PROBABILITY=.525` compares combined weekly volume/profit policies and
labels all evaluated seasons as exposed research. New standard runs retain checksummed per-cutoff
`lock_training.parquet` data; CLI `--training-history score-holdout --variants symmetric_residual_0.2` replays
those exact calibration inputs. Older caches support weekly-history research, not identical production calibration
history. The [backtesting guide](../../../../../docs/expected-points-backtesting.md#lightweight-lock-method-replay)
documents controls, provenance, uncertainty and mandatory production-path comparisons.
