# How the NFL model works

The model starts with a simple question: **how many points should each team score in this matchup?** Those two expected scores produce a projected game score, spread, and total. The model then compares its forecast with the available market lines to choose a side and a total for every game.

## From expected scores to picks

If the model projects San Francisco to win by 4.6 points while the market offers San Francisco +1.5, the displayed spread pick is **San Francisco +1.5**. The number shown first is always the line a bettor could act on; the underlying model projection is available in the game details.

Totals work the same way. A model total above the market line produces an Over pick, while a lower projection produces an Under pick.

## What powers the forecast

The score model is trained on play-by-play and game-level data beginning in 2010. Its most useful inputs include:

- Expected Points Added per play, split by offense, defense, passing, and rushing
- Success rate, which measures whether a play gained enough yardage for its down and distance
- Recent team form using smoothed moving averages
- Starting quarterback performance
- Rest, venue, and matchup context
- Available market information

The model predicts home and away scoring separately instead of directly predicting a betting outcome. This keeps the expected score as the foundation for both the spread and total.

## Pick win probability

A second model studies where the score model has historically been successful or unsuccessful against the market. It assigns a probability to the selected spread and total.

This is the probability of **that pick hitting**, not the probability of a team winning the game outright. A large difference between the projection and market line does not automatically mean high confidence; the probability model may recognize similar situations where the score model struggled.

## What makes a lock

A lock is a pick that passes additional qualification rules. It must have enough separation from the market, clear the required probability threshold, and rank among the strongest opportunities in the current slate.

“Lock” describes the model's highest-conviction category. It is not a guarantee.

## When predictions update

Predictions can change as new information enters the model. The NFL workflow normally refreshes during the week and again near kickoff. Once a game reaches its kickoff protection window, the saved prediction is preserved rather than rewritten.

## Glossary

**Expected Points Added (EPA):** The change in expected points caused by a play, accounting for field position, down, distance, and situation.

**Success Rate:** The percentage of plays that gain enough yardage relative to down and distance. The model uses 40% of yards-to-go on first down, 60% on second down, and a conversion on third or fourth down.

**Model spread:** The margin implied by the model's expected scores.

**Market line:** The current spread or total used to determine the actionable pick.

**Pick win probability:** The model's estimated chance that the displayed spread or total pick is correct.

**Lock:** A pick that passes the model's extra confidence and edge requirements.
