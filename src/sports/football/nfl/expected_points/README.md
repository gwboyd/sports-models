# NFL Expected Points Model

The visitor-facing, copyedited explanation rendered by the frontend is maintained in
`frontend/content/nfl-how-it-works.md`. This README remains the source for developer and operational workflow details.

The model predicts **expected points scored** for each NFL team in a game, getting to a predicted score from which a spread and total can be derived. We then compare those spreads and totals to the Vegas prices to make the picks each week. 

Score and confidence validation boundaries require more than eight hours between training and validation kickoffs,
matching the historical replay assumption for result availability. The split retains all games in one of the two
partitions. Historical cutoff fitting applies the same lag; exact result publication times are not reconstructed.

The score model uses play-by-play data beginning in 2010. Market-implied points anchor its forecast; EPA, success rate, rest, and market/context inputs help estimate a regularized correction.

Starting-quarterback ratings and actual roof status are excluded from the score recipe because the available feed does not preserve reliable pregame snapshots for those fields. Notebook diagnostics retain those columns.

`build_pregame_quarterback_metrics` constructs diagnostic quarterback history in kickoff order and keeps the existing
`ewma_qbr`/passer-rating feature names. Debut and unknown starters remain missing; never fill from a full-dataset
average of first starts. Score ridge preprocessing fits imputation and scaling only inside its training partition. Legacy confidence
classifiers also fit imputation chronologically; residual probability heads use score errors instead of quarterback
inputs. The September 2026 production audit removed the old future-dependent fallback;
the saved NFL 2.0 comparison predates that fix. A fresh performance comparison was explicitly waived by the user,
so do not attribute those measured gains to the final quarterback-history recipe.

The existing EPA and success-rate feature names now contain opponent-adjusted values. Before each game's kickoff,
the shared football transform fits ridge-regularized offense and defense effects only from results available at
the snapshot, requiring kickoff plus eight hours to precede that snapshot. It reassesses
each prior team performance against the defense or offense it faced, and then applies the existing moving-average
calculation. The adjustment therefore adds no future-game information and leaves downstream score and confidence
schemas unchanged. Its ridge penalty and cross-season carryover are recipe parameters for backtesting. Ratings use
the first kickoff of each game week by default, which excludes all same-week outcomes; the more expensive exact
kickoff mode remains available for evaluation.
In a notebook or Papermill run, set `opponent_adjustment_config` to partial overrides of the league defaults, such as
`{"ridge_alpha": 10.0, "season_carryover": 0.25}`; CFB additionally accepts
`{"fcs_policy": "pooled"}`.

The NFL recipe uses half-strength correction (`adjustment_strength=0.5`), selected on 2023–2024
screening weeks and evaluated on the standard 2023–2025 comparison recorded in
`.backtests/expected_points/reports/expected-points-2.0-evaluation.md` (local, Git-ignored).
`adjustment_strength` scales the opponent correction from zero (unadjusted history) to one (full correction); the moving-average calculation is unchanged. `excluded_seasons` removes listed seasons from both rating estimation and team efficiency history. A zero carryover with no current-season observations uses a neutral correction instead of attempting a zero-weight fit. These controls do not remove score-training games.

After picks are made, separate spread and total probability heads estimate the chance of a resolved bet winning
from out-of-time score errors. The 2.2 recipe uses the symmetric error distribution with trust 0.2 and ranks eligible Locks by
estimated win probability under a top-three weekly policy. The selection minimum and value limitations are described below.

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

Completed games are sorted by kickoff. The latest 20% form an outer score-model holdout. One chronological
validation split inside the earlier 80% selects ridge alpha and relative shrinkage of the team-feature group.
Preprocessing, including American-odds conversion, training-only imputation/scaling, and categorical encoding,
remains inside the fitted score pipeline. The score target is the residual from market-implied team points.

Before each outer-holdout week, the selected estimator refits on results available then without another GridSearch.
These weekly out-of-time forecasts train separate margin/total error distributions. The confidence-health tail
uses heads fitted only on earlier holdout forecasts. Finally, the selected score estimator refits on all eligible
completed games for production. The generic static holdout remains available for older recipe configurations.

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

Author NFL release notes in `UNRELEASED.md`; an empty draft means no unprepared changes, not necessarily no prepared
release. Do not add frontend, API-output, infrastructure, or database-only changes to model notes. Use a `#` title,
a plain-language `## Public Summary`, and `## Changes` with 2–3 concise bullets giving meaningful technical detail
(estimator, chronology, shrinkage, thresholds, or bet accounting). Optionally add one baseline-comparison bullet with
verified results for the actual recipe, the period/population, assumptions, and material uncertainty or selection
limitations. Keep complete evidence in Git-ignored `.backtests/expected_points/reports/`; omit source paths, commands,
experiment ledgers, and separate evaluation/internal headings. See the root README for the complete writing guidance.

Before committing/merging a model release, run `make prepare-model-release` on the feature branch. It reads
registered versions, prompts for major/minor for each populated NFL/CFB draft, and confirms before copying the exact
notes to `releases/vN.N.md`, clearing the drafts, and updating root `model-versions.json`. Commit these files and the
updated whole-model How It Works pages with the implementation, then merge once. Preparation writes no Supabase rows
and does not deploy. For any revision to unpublished prepared notes, including wording-only edits, copy the complete
notes back into UNRELEASED.md and edit there; do not hand-edit the archive or manifest. Review How It Works and update
it if behavior changes, then rerun prep. After confirmation it amends that same
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

The score model uses EPA per play and success rate, split across offense/defense and passing/rushing, alongside market and matchup context.

**EPA (Expected Points Added):** Measures the change in expected points on a play, accounting for the game situation before and after it.

**Success Rate:** "Success" is a tightly defined binary metric that determines if a play is successful or not dependent on the situation the offense is in. I used a custom success rate formula for this model that is as follows:

- On first down, gain at least 40% of the yards needed.
- On second down, gain at least 60% of the yards needed.
- On third or fourth down, reach the first-down marker.

In the **charts** section, there should be some graphics that show Offensive and Defensive EPA metrics for teams as the model currently sees them. These can be helpful if you are trying to peel back the curtain and understand why the model made a certain pick.

### Score adjustment weights

The ridge score model adds linear matchup corrections to market-implied points and penalizes large coefficients.
Its coefficients depend on training-only scaling and correlated inputs; they should not be interpreted as causal
feature importance. The old tree-feature-importance chart does not describe this score estimator.

### Moving Averages for Metrics

Play-level metrics are aggregated to games, adjusted for opponents, and smoothed with exponentially weighted
moving averages. EPA uses a span equal to the larger of ten games and the target week number; success rate uses
a ten-game span. These spans control decay rather than imposing a hard ten-game cutoff. Earlier games, including
previous seasons, can still contribute with diminishing weight.

Below is an example of the various calculations for the Dallas Cowboys. It should be fairly obvious why opt to smoothe out the metrics so much.

![Dynamic Window Example](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/dynamic_window_example.png)

## Determining what Picks to Play

The 2.2 recipe separates score prediction from bet selection. Both markets use `q=.5+.1*sign(e)*F(abs(e))`, where
`e` is the execution advantage in the selected pick's direction and `F` is the market-specific absolute-error CDF
from the weekly chronological score holdout. Symmetric errors are an assumption, not proof of calibration.
The shared selector ranks both markets by resolved win probability, then expected units, edge and stable game
keys. NFL selects its top three supported picks, even below estimated break-even at -110; extras are disabled.
Probabilities and expected units are unchanged. Model readiness, quote support and nonnegative execution edge
remain mandatory, so insufficient supported opportunities can leave fewer than three picks. Neither market has
a separate allocation cap. Started saved Locks consume
capacity even when absent from refreshed predictions and retain their original version. Exact projected-total
ties choose Over with zero edge and, for a ready residual head, q=0.5; actual scoring pushes remain pushes.
See the public methodology for the complete explanation and limitations.

## Misc

### Power Rankings

A separate LightGBM win classifier uses team-efficiency metrics, without odds, rest or other matchup context,
to simulate every team against every other team at home and away. A team's ranking averages its win probabilities
across those 62 hypothetical games. These simulations use current-week form, or next-week form for a team on a bye.
The power-ranking classifier tunes against one chronological train-before-validation split; it is supplemental chart
logic and does not feed the expected-points picks or betting probability heads.

### Ideas for the future

- Add announced starters and player-specific data with reliable pregame timestamps for injuries/trades
- Include explosiveness data 
- Add some 3rd down specifc metrics data
- Include some position group specific data to better catch matchup advantages
- Evaluate rookie and unknown-player handling when a pregame player-data source is available
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

### Adjustable weekly selection

`NFLExpectedPointsRecipe` owns independent `minimum_weekly_locks=3`,
`maximum_weekly_locks=3` and `extra_lock_min_probability=None` settings. The minimum is also the
normal selection count; the maximum caps extras. To enable NFL extras, set a probability threshold and raise its
maximum. Changing the count or thresholds is a prediction-affecting recipe change requiring a standard comparison
and release preparation. The notebook and production path both consume `build_config()`.
Shared `PlayThresholds.min_combined_plays` maps to `LockPolicy.min_locks_per_week`; zero retains the legacy
probability/EV floors and expected-units ranking. A positive minimum uses probability-first ranking and waives
only the probability/EV floors until that minimum is filled. Started selections consume both minimum and capacity.

## Lightweight Lock replay

Use `make replay-expected-points-locks LEAGUE=nfl` to anchor to the deployed release's latest matching local
standard score run; `LOCK_SOURCE=version:2.0` or `artifact:<run>` pins another source. No expected-score refit is
needed. `LOCK_SCREEN_CADENCE=1 LOCK_MIN_PROBABILITY=.525` compares combined weekly volume/profit policies and
labels all evaluated seasons as exposed research. New standard runs retain checksummed per-cutoff
`lock_training.parquet` data; CLI `--training-history score-holdout --variants symmetric_residual_0.2` replays
those exact calibration inputs. Older caches support weekly-history research, not identical production calibration
history. The [backtesting guide](../../../../../docs/expected-points-backtesting.md#lightweight-lock-method-replay)
documents controls, provenance, uncertainty and mandatory production-path comparisons.

The NFL confidence head retains the 2.1 trust 0.2 discount (80% shrinkage toward 50%). The top-three rule can select negative-EV picks; it does not change this conservative probability estimate or establish profitability.
