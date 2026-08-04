# NFL Expected Points Model

The model predicts **how many points each NFL team should score in a matchup**. Those expected scores produce a projected final score, spread, and total. The model then compares its forecast with the available market lines to select a side and total for every game.

The score model uses play-by-play and game-level data beginning in 2010. Expected Points Added (EPA) per play and Success Rate have been the most useful predictors, alongside starting-quarterback performance, rest, venue, matchup context, and available market information.

After the picks are made, separate spread and total classifiers study the model's historical predictions against the market. They identify the situations in which the score model has been most and least reliable and estimate the probability that each selected pick will hit. Those probabilities, together with additional qualification rules, determine the week's **Locks**.

## From expected scores to picks

If the model projects San Francisco to win by 4.6 points while the market offers San Francisco +1.5, the displayed spread pick is **San Francisco +1.5**. The number shown first is always the actionable market line; the underlying model projection remains available in the game details.

Totals work the same way. A projected total above the market line produces an Over pick, while a lower projection produces an Under pick.

## Prediction update cadence

The model can be updated whenever new information is available, but its scheduled NFL refreshes are:

- Wednesday at 12:00 AM Eastern
- Thursday at 7:20 PM Eastern
- Sunday at 12:00 PM Eastern

The workflow does not replace a saved prediction inside the kickoff protection window. Its final pregame prediction is preserved once the game is within 30 minutes of kickoff.

## Features

The model is primarily powered by EPA per play and Success Rate, split across offense, defense, passing, and rushing. These efficiency metrics are more predictive than raw yardage or points alone.

**EPA (Expected Points Added):** Measures how much a play changed the offense's expected points, accounting for field position, down, distance, and game situation. High-impact plays receive more credit than routine gains.

**Success Rate:** A binary measure of whether a play gained enough yardage for its situation. This model defines a successful play as:

- More than 40% of the required yards on first down
- More than 60% of the required yards on second down
- A conversion on third or fourth down

The [Model Insights](/models/nfl/insights) page shows the current offensive and defensive EPA views, power rankings, and other live model outputs.

### Feature importance

The expected-points model uses a gradient-boosted framework. It does not assign one simple linear weight to every input, but feature importance still provides a useful view of which inputs contribute most to its forecasts.

![Model feature importances](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/feature_importance.png)

Feature importance can change as new games enter the training data, although the model has now seen enough seasons that the broad ordering is relatively steady. Efficiency metrics have generally remained more useful than information originating from market data.

The `pred_team` feature reflects the value the model assigns to venue and home-field context.

### Moving averages for team metrics

Advanced metrics begin at the play level and are aggregated to games. The model then uses exponentially weighted moving averages to reduce week-to-week noise while giving more influence to recent performance.

The window is dynamic. It generally uses the current season, but before Week 10 it includes the most recent ten games and reaches into the previous season when necessary.

The chart below shows why smoothing is useful: raw game-level performance is volatile, while the moving averages preserve the underlying trend.

![Dynamic moving-average window example](https://nfl-metrics.s3.us-east-1.amazonaws.com/charts/dynamic_window_example.png)

## Pick win probability and Locks

Separate classifiers observe the kinds of spread and total predictions the score model has historically handled well—and the situations in which it has struggled against the market.

The classifiers train on predictions for games the score model did not train on. This prevents the confidence model from grading artificially easy, already-seen forecasts. There is a tradeoff between the amount of training data available to the expected-points model and the out-of-sample predictions available to the probability classifiers.

**Why can the probability be modest when the model and market are far apart?** Pick probability is not calculated from the size of that difference alone. The classifier may recognize similar historical situations in which a large apparent edge did not hold up.

A **Lock** is a pick that clears the required probability, model-edge, and slate-ranking rules. It is the model's highest-conviction category, not a guarantee.

Win probability always refers to the displayed spread or total pick hitting. It is not the probability that a team wins the game outright.

## Power rankings

Power rankings are produced from the model's team-strength metrics without odds, rest, or other game-specific context. A separate win classifier simulates every team playing every other team at home and away.

Each team's simulated win percentage is the average of its win probabilities across those 62 hypothetical games. The simulations use the team's current form, or its next-week form when the team is on a bye, so the rankings answer: **How would every team compare if they all played today?**

## Ideas for the future

- Add more player-specific data beyond quarterbacks to better account for injuries and trades
- Adjust efficiency metrics for opponent difficulty
- Include explosiveness and more third-down-specific features
- Add position-group metrics to identify matchup advantages
- Improve how rookie and low-sample players are represented
- Reduce dependence on market information where predictive quality allows
- Add opening lines and public-positioning data to better measure changes in market sentiment

## Glossary

**Model spread:** The scoring margin implied by the model's expected scores.

**Market line:** The current spread or total used to determine the actionable pick.

**Pick win probability:** The model's estimated chance that the displayed spread or total pick is correct.

**Lock:** A pick that passes the model's additional confidence, edge, and slate-ranking requirements.
