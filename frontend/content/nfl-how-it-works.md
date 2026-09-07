# NFL Expected Points Model

This page describes the **2.1 methodology**. Each saved pick keeps the model version that produced it, so historical results can include earlier versions.

The model predicts **how many points each NFL team should score in a matchup**. Those expected scores produce a projected final score, spread, and total. The model then compares its forecast with the available market lines to select a side and total for every game.

The score model uses play-by-play and game-level data beginning in 2010. Expected Points Added (EPA) per play and Success Rate have been the most useful predictors, alongside starting-quarterback performance, rest, venue, matchup context, and available market information.

After the picks are made, separate spread and total probability models use unseen-game score errors to estimate betting confidence. Estimates are strongly pulled toward an even chance. Locks are selected using those probabilities and expected profit, with roughly two or three combined opportunities as the usual aim rather than a forced quota.

## How the score model learns

A gradient-boosted model learns the relationship between matchup inputs and the points each team scores. Each game contributes a home-team and an away-team scoring example, and both examples stay together when data is split for validation.

Training follows time order. The model selects its settings using earlier games and a later validation period, then makes predictions for a separate later holdout. Those unseen-game predictions supply the training examples for the spread and total probability models. After selecting the score-model settings, it refits on all eligible completed games before predicting the upcoming slate.

Starting-quarterback performance uses a moving average of that player's earlier starts, ordered by kickoff. When there is no earlier start, the quarterback-history input remains missing rather than borrowing information from later games. The score model handles missing values directly.

## From expected scores to picks

If the model projects San Francisco to win by 4.6 points while the market offers San Francisco +1.5, the displayed spread pick is **San Francisco +1.5**. The number shown first is always the actionable market line; the underlying model projection remains available in the game details.

Totals work the same way. A projected total above the market line produces an Over pick, while a lower projection produces an Under pick.

## Prediction update cadence

During the active season, scheduled refreshes run daily at 5:00 AM Eastern. On game days, additional refreshes are scheduled one hour before the first game in each occupied kickoff window: before 2:00 PM, 2:00–6:59 PM, and 7:00 PM or later, all Eastern. Offseason refreshes run weekly once a future slate is available.

A saved prediction is preserved when its game enters the 30-minute kickoff protection window. The model never creates a new prediction at or after kickoff. Game times and update timestamps on the site display in your device's timezone.

## Features

The model is primarily powered by EPA per play and Success Rate, split across offense, defense, passing, and rushing. These efficiency metrics are more predictive than raw yardage or points alone.

**EPA (Expected Points Added):** Measures how much a play changed the offense's expected points, accounting for field position, down, distance, and game situation. High-impact plays receive more credit than routine gains.

**Success Rate:** A binary measure of whether a play gained enough yardage for its situation. This model defines a successful play as:

- At least 40% of the required yards on first down
- At least 60% of the required yards on second down
- A conversion on third or fourth down

The [Model Insights](/models/nfl/insights) page shows the current offensive and defensive EPA views, power rankings, and other live model outputs.

### Opponent adjustment with ridge regression

A strong rushing game against an excellent run defense tells us more than the same performance against a weak one. The model adjusts EPA and success rate for the quality of the opponents faced, separately for each offensive and defensive passing/rushing metric.

For each metric, a **ridge regression** estimates offensive strength, defensive strength, and venue effects from earlier games. The relationship is:

**Observed efficiency ≈ average efficiency + offense effect + opposing defense effect + venue effect.**

The offense and defense estimates are fitted together across the schedule. This handles the apparent circularity: we do not need to decide how good every defense is before estimating every offense. Ridge adds a penalty for large effects, pulling estimates toward neutral when the data offers little support for a strong rating. It helps keep limited samples from producing extreme conclusions.

For an offensive performance, the model removes half of the estimated opposing defense effect. For a defensive performance, it removes half of the opposing offense effect. For example, if an opponent is estimated to inflate offensive EPA by 0.10 points per play, an observed 0.20 becomes 0.15 after the half-strength correction. This is an illustration, not a current team rating.

The adjusted game performances then pass through the moving averages below. The ridge regression prepares the efficiency inputs; a separate gradient-boosted score model turns those inputs and the other matchup information into expected scores. Starting-quarterback performance is still a separate input and does not receive this opponent adjustment.

### Early-season ratings and information timing

Ratings use only games played before the opening kickoff of the target game week. As new weeks arrive, earlier performances are reassessed using the opponent estimates available then; future games never enter an earlier week's estimate.

Previous seasons help when current-season evidence is scarce. In the opponent-rating fit, each season back halves a game's weight: a prior-season game receives half the weight of a current-season game, and a game two seasons back receives a quarter. Ridge also limits unsupported effects. An opponent with no estimated effect receives a neutral correction. This cannot fully account for roster turnover or guarantee an accurate early-season rating.

### Feature importance

The expected-points model uses a gradient-boosted framework. It does not assign one simple linear weight to every input, but feature importance still provides a useful view of which inputs contribute most to its forecasts.

![Model feature importances](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/feature_importance.png)

Feature importance can change as new games enter the training data, although the model has now seen enough seasons that the broad ordering is relatively steady. Efficiency metrics have generally remained more useful than information originating from market data.

The model also distinguishes home and away teams to account for venue and home-field context.

### Moving averages for team metrics

Advanced metrics begin at the play level and are aggregated to games. The model then uses exponentially weighted moving averages to reduce week-to-week noise while giving more influence to recent performance.

EPA uses a smoothing span equal to the larger of ten games and the current week number. Success-rate and quarterback metrics use a ten-game span. Recent games receive more weight, while older games can still contribute, including games from previous seasons. These are gradual weighting schemes rather than hard cutoffs on the number of games used.

The chart below shows why smoothing is useful: raw game-level performance is volatile, while the moving averages preserve the underlying trend.

![Dynamic moving-average window example](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/dynamic_window_example.png)

## Pick win probability and Locks

The probability layer uses forecasts for games outside the score model's training sample. For spreads it measures errors in the projected margin; for totals it measures errors in the projected combined score. These chronological holdout errors are kept separate from the final score-model refit.

Let `e` be the absolute gap between the model prediction and the displayed betting line, and `F(e)` the fraction of earlier absolute score errors smaller than that gap. The estimated resolved-bet win probability is `q = 0.5 + 0.1 × F(e)`. This assumes symmetric score errors and discounts their implied advantage by 80%, keeping estimates between 50% and 60%. Symmetry and the fixed discount are modeling assumptions, not proof of perfect calibration. Fewer than 100 resolved examples or fewer than 25 wins or losses triggers a shrunken historical-rate fallback with no Locks.

A Lock needs at least a 52.5% estimated resolved-win probability and positive expected profit at assumed -110 pricing. A one-unit win earns about 0.909 units and a loss costs one unit. With estimated push probability `s`, expected profit is `(1-s) × [0.909 × q - (1-q)]`. Push estimates use earlier integer-line outcomes. Actual sportsbook prices may differ and can change the value of a bet. The probability refers to the displayed betting pick, not whether a team wins outright.

Candidates are ranked by expected profit. The normal combined limit is three per NFL week, with up to two spreads and one total. Additional candidates may exceed these normal limits only at a stronger 55% probability floor; there is no separate hard ceiling of five. This is not a weekly minimum: thin slates can have fewer or no Locks. Preserved picks near kickoff retain their original version and Lock status and count toward weekly capacity.

A **Lock** is a selective category, not a guarantee. Historical comparisons guided these rules, but repeated experimentation and limited samples mean future profitability is not established.

## Power rankings

Power rankings are produced from the model's team-strength metrics without odds, rest, or other game-specific context. A separate win classifier simulates every team playing every other team at home and away.

Each team's simulated win percentage is the average of its win probabilities across those 62 hypothetical games. The simulations use the team's current form, or its next-week form when the team is on a bye, so the rankings answer: **How would every team compare if they all played today?**

## Ideas for the future

- Add more player-specific data beyond quarterbacks to better account for injuries and trades
- Include explosiveness and more third-down-specific features
- Add position-group metrics to identify matchup advantages
- Improve how rookie and low-sample players are represented
- Reduce dependence on market information where predictive quality allows
- Add opening lines and public-positioning data to better measure changes in market sentiment

## Glossary

**Model spread:** The scoring margin implied by the model's expected scores.

**Market line:** The current spread or total used to determine the actionable pick.

**Pick win probability:** The model's estimated chance that the displayed spread or total pick is correct.

**Lock:** A spread or total pick that passes the win-probability, positive-value, market-support, and combined weekly-selection requirements.
