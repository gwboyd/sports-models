# CFB Expected Points Model

This page describes the **2.1 methodology**. Each saved pick keeps the model version that produced it, so historical results can include earlier versions.

The model predicts **how many points each college football team should score in a matchup**. Those expected scores produce a projected spread and total, which are compared with sportsbook lines to select a side and an Over or Under. Separate probability models estimate how likely each displayed pick is to hit.

College football teams face very different schedules. The model accounts for opponent strength, combines several measures of offensive and defensive efficiency, and trains on games beginning in **2018**. The **2020 COVID season is excluded** from training and team-strength calculations. Efficiency history begins in 2017 to give the earliest training games background information.

## From expected scores to picks

Suppose the model projects a team to win by seven points and the market spread is that team -3.5. The spread pick is **that team -3.5**. If the projected total is 55 and the market total is 51.5, the total pick is **Over 51.5**. The displayed line is the line used to grade the pick.

The model uses the median of available eligible sportsbook lines as its market reference. It chooses a direction against that consensus, then looks for a favorable execution line within the largest group of providers whose quotes are within two points of one another. This limits the influence of an isolated outlier quote.

A game must include at least one FBS team and have an eligible sportsbook spread and total to receive a prediction. FBS–FCS games can qualify. A market with only one supporting provider can appear as a pick but cannot qualify as a Lock; Locks require corroborating quotes from at least two providers.

## Team efficiency and matchup context

The score model uses three measures for both offense and defense:

- **Predicted Points Added (PPA):** The scoring value associated with plays, based on expected points in the game situation.
- **Success rate:** How consistently an offense produces successful plays, or a defense prevents them.
- **Explosiveness:** The scoring value of successful plays, capturing how much impact those plays have.

These measures describe different aspects of performance: overall value, consistency, and impact. They are adjusted for opponents and smoothed across past games. The model also uses pregame Elo ratings, whether the matchup is a conference game, the day of the week, and points implied by the market. Elo summarizes team strength from past game results.

## How the score model learns

A gradient-boosted model predicts each team's points from the efficiency and matchup inputs. A game contributes both a home-team and an away-team scoring example; the two examples remain together in chronological training and validation splits.

The model tunes its settings on earlier games with a later validation period, then predicts a separate later holdout. Those unseen-game predictions teach the spread and total probability heads how reliable the model's picks are. The score model then refits on all eligible completed games with its selected settings before forecasting the upcoming slate.

The score model handles missing feature values directly. The probability layer uses valid historical score errors and resolved outcomes; it does not fill a past feature with an average calculated from future games.

## Opponent adjustment with ridge regression

The same offensive performance means something different against a strong defense than against a weak one. For each efficiency metric, a **ridge regression** estimates offense and defense effects together, while accounting for venue and the FBS/FCS distinction:

**Observed efficiency ≈ average efficiency + offense effect + opposing defense effect + venue effect.**

This is a joint fit across earlier games. It handles the circular relationship between offense and defense without first assuming that either set of ratings is already correct. Ridge penalizes large effects, so estimates with little supporting evidence stay closer to neutral.

The model removes **half** of the estimated opponent effect from a historical performance. An offense gets more credit for producing against a difficult defense and less against an easy one; a defense is assessed against the offenses it faced. Applying half the correction keeps observed performance influential when opponent estimates are uncertain.

For illustration, if an opposing defense is estimated to inflate PPA by 0.10 points per play, an observed 0.20 becomes 0.15 after adjustment. The resulting efficiency history is smoothed before it enters the score model.

Ridge is the opponent-rating step. A separate **gradient-boosted score model** learns how these inputs combine to predict each team's points, including relationships that are not captured by a single linear weight.

### FBS and FCS opponents

FBS teams receive individual offense and defense estimates. FCS teams share a pooled offense estimate and a pooled defense estimate because the data available for individual FCS teams is more limited. The model learns these effects from the available history rather than assigning every FCS game a fixed penalty.

Pooling lets FCS matchups contribute without treating a handful of observations as a reliable individual rating. It also means the model does not distinguish stronger and weaker FCS teams through these opponent effects. That is a limitation of this version.

### Beginning of the season

The model carries earlier seasons into the opponent-rating fit with decreasing weight. Each season back halves a game's weight: 0.5 for the previous season and 0.25 for two seasons back, relative to a current-season game. Excluded 2020 games never contribute, and the discount follows calendar seasons even across that gap.

Ridge pulls poorly supported effects toward neutral, and an opponent without an estimated effect receives a neutral correction. This gives the model a starting point before many current-season games exist, but it cannot fully capture changes from the transfer portal, coaching moves, or player departures. Player-specific inputs are a future direction rather than part of this adjustment.

### Recent form and information timing

Adjusted game performances feed exponentially weighted moving averages, which give recent games more influence. The smoothing span is the larger of ten games and the target week number. Older games can still contribute; ten is not a hard cutoff on the number of games used.

Opponent ratings and efficiency history use only games before the opening kickoff of the target game week. Earlier performances are reassessed as new weeks provide more evidence about opponents. No future game is used to construct an earlier week's efficiency inputs.

## Pick win probability and Locks

The spread probability layer uses forecasts for games outside the score model's training sample and measures errors in the projected margin. These chronological holdout errors are kept separate from the final score-model refit. Total probabilities use resolved outcomes from that same holdout.

Let `e` be the absolute gap between the model prediction and the displayed betting line, and `F(e)` the fraction of earlier absolute score errors smaller than that gap. The estimated resolved-bet win probability is `q = 0.5 + 0.1 × F(e)`. This assumes symmetric score errors and discounts their implied advantage by 80%, keeping estimates between 50% and 60%. Symmetry and the fixed discount are modeling assumptions, not proof of perfect calibration. Fewer than 100 resolved examples or fewer than 25 wins or losses triggers a shrunken historical-rate fallback with no Locks.

A Lock needs at least a 52.5% estimated resolved-win probability and positive expected profit at assumed -110 pricing. A one-unit win earns about 0.909 units and a loss costs one unit. With estimated push probability `s`, expected profit is `(1-s) × [0.909 × q - (1-q)]`. Push estimates use earlier integer-line outcomes. Actual sportsbook prices may differ and can change the value of a bet. The probability refers to the displayed betting pick, not whether a team wins outright.

The score-error method applies to spreads. Total-pick probabilities use a strongly shrunken historical hit rate, and total Locks are disabled in this version. Candidates are ranked by expected profit. The normal combined weekly limit is three, currently all spreads. Additional spreads may exceed that limit only at a stronger 55% probability floor; there is no separate hard ceiling of five. At least two corroborating real sportsbook quotes are still required. This is not a weekly minimum: thin slates can have fewer or no Locks. Preserved picks near kickoff retain their original version and Lock status and count toward weekly capacity.

A **Lock** is a selective category, not a guarantee. Historical comparisons guided these rules, but repeated experimentation and limited samples mean future profitability is not established.

## Prediction updates and results

During the active season, scheduled refreshes run daily at 5:00 AM Eastern. On game days, additional refreshes occur one hour before the first game in each occupied kickoff window: before 2:00 PM, 2:00–6:59 PM, and 7:00 PM or later, all Eastern. Offseason refreshes run weekly once a future slate is available.

Saved picks are preserved when their games enter the 30-minute kickoff protection window. No new prediction is created at or after kickoff. The site displays kickoff and update times in your device's timezone.

Completed games are graded on the next successful model update. [Results](/models/cfb/results) show historical spread and total performance. A pick keeps the version that generated it even when a newer model later grades the result.

## Glossary

**Model spread:** The scoring margin implied by the expected scores.

**Market reference:** The median eligible sportsbook line used to choose the pick's direction.

**Execution line:** The displayed sportsbook line used to measure the edge and grade the pick.

**Pick win probability:** The estimated chance that the selected spread or total pick hits.

**Lock:** A pick that passes the model's win-probability, positive-value, sportsbook-support, and combined weekly-ranking rules. Only spread picks are eligible in this version.
