# NFL Expected Points Model

This page describes the **2.2 methodology**. Each saved pick keeps the model version that produced it, so historical results can include earlier versions.

The model predicts **how many points each NFL team should score in a matchup**. Those expected scores produce a projected final score, spread, and total. The model then compares its forecast with the available market lines to select a side and total for every game.

The score model uses play-by-play and game-level data beginning in 2010. Market-implied points provide its starting forecast. Expected Points Added (EPA), success rate, rest, and matchup context help it learn adjustments from earlier games.

After the picks are made, separate spread and total probability models use unseen-game score errors to estimate betting confidence. Locks are selected using those probabilities and expected profit, with at most three combined opportunities per week and no forced quota.

## How the score model learns

A ridge-regression score model starts with the points implied by the market spread and total, then learns a correction from the matchup inputs. For a home spread `s` and total `t`, the market-implied scores are `(t-s)/2` for the home team and `(t+s)/2` for the away team. The final forecast is **market-implied points + learned adjustment**, floored at zero.

Ridge penalizes large coefficients so weak or noisy evidence produces smaller adjustments. The model selects the penalty and how strongly to shrink the team-performance inputs using one chronological validation split inside the earlier training history. It can reduce the entire team-performance group to zero when that earlier validation favors market and context inputs. These choices are made again as training history grows; they are not chosen from the upcoming games.

Each game contributes a home-team and an away-team scoring example. Both examples stay together in every chronological split, and team-specific inputs rotate with the team being predicted. Numerical inputs use medians and scales fitted only on the training partition, with indicators for missing values. Categorical inputs use a separate missing category; an unseen category can be handled without learning from the prediction slate.

A separate, later score holdout supplies confidence-training forecasts. Before each historical week in that holdout, the score model refits on the games available then using the already selected settings. It does not retune on the holdout. Earlier holdout results can inform later weeks only after they become available. A later confidence-health tail checks the probability and Lock selection process. Historical fitting assumes results become available eight hours after kickoff, and split boundaries allow that gap. Finally, the score model refits on eligible completed games before forecasting the upcoming slate. Ordinary training also applies the eight-hour availability guard at the prediction timestamp; an earlier week label cannot admit a postponed or still-unavailable result.

## From expected scores to picks

If the model projects San Francisco to win by 4.6 points while the market offers San Francisco +1.5, the displayed spread pick is **San Francisco +1.5**. The number shown first is always the actionable market line; the underlying model projection remains available in the game details.

Totals work the same way. A projected total above the market line produces an Over pick, while a lower projection produces an Under pick.

## Prediction update cadence

During the active season, scheduled refreshes run daily at 5:00 AM Eastern. On game days, additional refreshes are scheduled one hour before the first game in each occupied kickoff window: before 2:00 PM, 2:00–6:59 PM, and 7:00 PM or later, all Eastern. Offseason refreshes run weekly once a future slate is available.

A saved prediction is preserved when its game enters the 30-minute kickoff protection window. The model never creates a new prediction at or after kickoff. Game times and update timestamps on the site display in your device's timezone.

## Features

EPA per play and success rate are split across offense, defense, passing, and rushing. They describe team performance alongside the market inputs, rather than treating raw yardage or previous points as a complete measure of team strength.

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

The adjusted game performances then pass through the moving averages below. The ridge regression prepares the efficiency inputs; the separate ridge score model uses those inputs and other matchup information to adjust market-implied points.

### Early-season ratings and information timing

Ratings use only games played before the opening kickoff of the target game week. As new weeks arrive, earlier performances are reassessed using the opponent estimates available then; future games never enter an earlier week's estimate.

Previous seasons help when current-season evidence is scarce. In the opponent-rating fit, each season back halves a game's weight: a prior-season game receives half the weight of a current-season game, and a game two seasons back receives a quarter. Ridge also limits unsupported effects. An opponent with no estimated effect receives a neutral correction. This cannot fully account for roster turnover or guarantee an accurate early-season rating.

### Other inputs and missing history

Rest, weekday, division matchups, and available betting information supply additional context. American odds are converted to implied probabilities before entering the linear model. Starting-quarterback ratings and roof status are excluded: the historical feed identifies quarterbacks from game participation and records actual retractable-roof decisions, rather than preserving those inputs as they were known before a prediction. The market remains an indirect source of personnel and weather information.

### Moving averages for team metrics

Advanced metrics begin at the play level and are aggregated to games. The model then uses exponentially weighted moving averages to reduce week-to-week noise while giving more influence to recent performance.

EPA uses a smoothing span equal to the larger of ten games and the current week number. Success-rate metrics use a ten-game span. Recent games receive more weight, while older games can still contribute, including games from previous seasons. These are gradual weighting schemes rather than hard cutoffs on the number of games used.

The chart below shows why smoothing is useful: raw game-level performance is volatile, while the moving averages preserve the underlying trend.

![Dynamic moving-average window example](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/dynamic_window_example.png)

## Pick win probability and Locks

The probability layer uses chronological forecasts for games that were outside the score model's training sample when predicted. It measures errors in the projected margin for spreads and in the projected combined score for totals. Each forecast is refreshed using the earlier history available at that historical week, keeping its own outcome out of training.

Let `e` be the model's advantage in the selected pick's direction at the displayed execution line, and `F(|e|)` the fraction of earlier absolute prediction errors smaller than that advantage. The estimated resolved-bet win probability is `q = 0.5 + 0.1 × sign(e) × F(|e|)`. A positive advantage increases confidence; an adverse execution quote lowers it below 50% and cannot qualify as a Lock. NFL retains an 80% shrinkage toward 50% rather than treating the full error-distribution estimate as certain. This assumes a symmetric error distribution that remains relevant to the next matchup. It is an approximation, not proof of perfect calibration. Fewer than 100 resolved examples or fewer than 25 wins or losses triggers a shrunken historical-rate fallback with no Locks.

A Lock needs at least a 52.5% estimated resolved-win probability and positive expected profit at assumed -110 pricing. A one-unit win earns about 0.909 units and a loss costs one unit. With estimated push probability `s`, expected profit is `(1-s) × [0.909 × q - (1-q)]`. Push estimates use earlier integer-line outcomes. Actual sportsbook prices may differ and can change the value of a bet. The probability refers to the displayed betting pick, not whether a team wins outright.

Candidates are ranked by expected profit. The hard combined limit is three per NFL week, with up to two spreads and one total. Higher confidence does not override these limits. Small model advantages can produce long stretches with few or no NFL Locks; the limit is not a quota. Preserved picks near kickoff retain their original version and Lock status and count toward weekly capacity.

A **Lock** is a selective category, not a guarantee. Historical comparisons guided these rules, but repeated experimentation and limited samples mean future profitability is not established.

## Power rankings

Power rankings are produced from the model's team-strength metrics without odds, rest, or other game-specific context. A separate win classifier simulates every team playing every other team at home and away.

Each team's simulated win percentage is the average of its win probabilities across those 62 hypothetical games. The simulations use the team's current form, or its next-week form when the team is on a bye, so the rankings answer: **How would every team compare if they all played today?**

## Ideas for the future

- Add announced starters and player-specific information with reliable pregame timestamps to account for injuries and trades
- Include explosiveness and more third-down-specific features
- Add position-group metrics to identify matchup advantages
- Improve how rookie and low-sample players are represented
- Reduce dependence on market information where predictive quality allows
- Add opening lines and public-positioning data to better measure changes in market sentiment

## Limits of historical evaluation

Walk-forward comparisons recreate weekly decisions using currently available historical feed values. They cannot recover every intraday line, injury update, or later data correction exactly as it appeared at the time. Repeated experiments on the same seasons also create selection risk. Score-error improvements and past winning records support model development, but neither guarantees that the next season will improve or be profitable.

## Glossary

**Model spread:** The scoring margin implied by the model's expected scores.

**Market line:** The current spread or total used to determine the actionable pick.

**Pick win probability:** The model's estimated chance that the displayed spread or total pick is correct.

**Lock:** A spread or total pick that passes the win-probability, positive-value, market-support, and combined weekly-selection requirements.
