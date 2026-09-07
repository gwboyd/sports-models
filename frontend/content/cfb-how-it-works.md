# CFB Expected Points Model

This page describes the **2.2 methodology**. Each saved pick keeps the model version that produced it, so historical results can include earlier versions.

The model predicts **how many points each college football team should score in a matchup**. Those expected scores produce a projected spread and total, which are compared with sportsbook lines to select a side and an Over or Under. Separate probability models estimate how likely each displayed pick is to hit.

College football teams face very different schedules. The model accounts for opponent strength, combines several measures of offensive and defensive efficiency, and trains on games beginning in **2018**. The **2020 COVID season is excluded** from training and team-strength calculations. Efficiency history begins in 2017 to give the earliest training games background information.

## From expected scores to picks

Suppose the model projects a team to win by seven points and the market spread is that team -3.5. The spread pick is **that team -3.5**. If the projected total is 55 and the market total is 51.5, the total pick is **Over 51.5**. The displayed line is the line used to grade the pick.

The model uses the median of available eligible sportsbook lines as its market reference. It chooses a direction against that consensus, then looks for a favorable execution line within the largest group of providers whose quotes are within two points of one another. This limits the influence of an isolated outlier quote.

A game must include at least one FBS team and have an eligible sportsbook spread and total to receive a prediction. FBS–FCS games can qualify. Nonpositive totals are invalid. A market with only one supporting provider can appear as a pick but cannot qualify as a Lock; Locks require corroborating quotes from at least two providers. If the supported quote cluster lies on the unfavorable side of the model projection, the consensus-selected pick remains visible but its negative execution edge prevents it from becoming a Lock.

## Team efficiency and matchup context

The score model uses three measures for both offense and defense:

- **Predicted Points Added (PPA):** The scoring value associated with plays, based on expected points in the game situation.
- **Success rate:** How consistently an offense produces successful plays, or a defense prevents them.
- **Explosiveness:** The scoring value of successful plays, capturing how much impact those plays have.

These measures describe different aspects of performance: overall value, consistency, and impact. They are adjusted for opponents and smoothed across past games. The model also uses pregame Elo ratings, whether the matchup is a conference game, the day of the week, and points implied by the market. Elo summarizes team strength from past game results. Its weekly snapshot uses only completed games available before the first kickoff of that week, including an eight-hour allowance for results. If a team plays twice in the same recorded week, both forecasts use that earlier snapshot; the first result cannot inform a forecast made before it happened.

## How the score model learns

A ridge-regression score model starts with the points implied by the market spread and total, then learns a correction from the matchup inputs. For a home spread `s` and total `t`, the market-implied scores are `(t-s)/2` for the home team and `(t+s)/2` for the away team. The final forecast is **market-implied points + learned adjustment**, floored at zero.

Ridge penalizes large coefficients so weak or noisy evidence produces smaller adjustments. The model selects the penalty and how strongly to shrink the team-performance inputs using one chronological validation split inside the earlier training history. It can reduce the entire team-performance group to zero when that earlier validation favors market and context inputs. These choices are made again as training history grows; they are not chosen from the upcoming games.

Each game contributes a home-team and an away-team scoring example. Both examples stay together in every chronological split, and team-specific inputs rotate with the team being predicted. Numerical inputs use medians and scales fitted only on the training partition, with indicators for missing values. Categorical inputs use a separate missing category; an unseen category can be handled without learning from the prediction slate.

A separate, later score holdout supplies confidence-training forecasts. Before each historical week in that holdout, the score model refits on the games available then using the already selected settings. It does not retune on the holdout. Earlier holdout results can inform later weeks only after they become available. A later confidence-health tail checks the probability and Lock selection process. Historical fitting assumes results become available eight hours after kickoff, and split boundaries allow that gap. Finally, the score model refits on eligible completed games before forecasting the upcoming slate. Ordinary training also applies the eight-hour availability guard at the prediction timestamp; an earlier week label cannot admit a postponed or still-unavailable result.

## Opponent adjustment with ridge regression

The same offensive performance means something different against a strong defense than against a weak one. For each efficiency metric, a **ridge regression** estimates offense and defense effects together, while accounting for venue and the FBS/FCS distinction:

**Observed efficiency ≈ average efficiency + offense effect + opposing defense effect + venue effect.**

This is a joint fit across earlier games. It handles the circular relationship between offense and defense without first assuming that either set of ratings is already correct. Ridge penalizes large effects, so estimates with little supporting evidence stay closer to neutral.

The model removes **half** of the estimated opponent effect from a historical performance. An offense gets more credit for producing against a difficult defense and less against an easy one; a defense is assessed against the offenses it faced. Applying half the correction keeps observed performance influential when opponent estimates are uncertain.

For illustration, if an opposing defense is estimated to inflate PPA by 0.10 points per play, an observed 0.20 becomes 0.15 after adjustment. The resulting efficiency history is smoothed before it enters the score model.

Opponent adjustment and score prediction are separate ridge fits. The first estimates the difficulty of past opponents; the second learns how the resulting matchup inputs adjust market-implied team points.

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

The probability layer uses chronological forecasts for games that were outside the score model's training sample when predicted. It measures errors in the projected margin for spreads and in the projected combined score for totals. Each forecast is refreshed using the earlier history available at that historical week, keeping its own outcome out of training.

Let `e` be the model's advantage in the selected pick's direction at the displayed execution line, and `F(|e|)` the fraction of earlier absolute prediction errors smaller than that advantage. The estimated resolved-bet win probability is `q = 0.5 + 0.5 × sign(e) × F(|e|)`. A positive advantage increases confidence; an adverse execution quote lowers it below 50% and cannot qualify as a Lock. This assumes a symmetric error distribution that remains relevant to the next matchup. It is an approximation, not proof of perfect calibration. Fewer than 100 resolved examples or fewer than 25 wins or losses triggers a shrunken historical-rate fallback with no Locks.

A Lock needs at least a 52.5% estimated resolved-win probability and positive expected profit at assumed -110 pricing. A one-unit win earns about 0.909 units and a loss costs one unit. With estimated push probability `s`, expected profit is `(1-s) × [0.909 × q - (1-q)]`. Push estimates use earlier integer-line outcomes. Actual sportsbook prices may differ and can change the value of a bet. The probability refers to the displayed betting pick, not whether a team wins outright.

The score-error method applies to both spreads and totals. Candidates are ranked together by expected profit, with a hard limit of three combined Locks per college football week and at most one total. Higher confidence does not override these limits. At least two corroborating real sportsbook quotes are required. Thin slates can have fewer or no Locks; the limit is not a quota. Preserved picks near kickoff retain their original version and Lock status and count toward weekly capacity.

A **Lock** is a selective category, not a guarantee. Historical comparisons guided these rules, but repeated experimentation and limited samples mean future profitability is not established.

## Prediction updates and results

During the active season, scheduled refreshes run daily at 5:00 AM Eastern. On game days, additional refreshes occur one hour before the first game in each occupied kickoff window: before 2:00 PM, 2:00–6:59 PM, and 7:00 PM or later, all Eastern. Offseason refreshes run weekly once a future slate is available.

Saved picks are preserved when their games enter the 30-minute kickoff protection window. No new prediction is created at or after kickoff. The site displays kickoff and update times in your device's timezone.

Completed games are graded on the next successful model update. [Results](/models/cfb/results) show historical spread and total performance. A pick keeps the version that generated it even when a newer model later grades the result.

## Limits of historical evaluation

Walk-forward comparisons recreate weekly decisions using currently available historical feed values. They cannot recover every intraday line, injury update, or later data correction exactly as it appeared at the time. Repeated experiments on the same seasons also create selection risk. Score-error improvements and past winning records support model development, but neither guarantees that the next season will improve or be profitable.

## Glossary

**Model spread:** The scoring margin implied by the expected scores.

**Market reference:** The median eligible sportsbook line used to choose the pick's direction.

**Execution line:** The displayed sportsbook line used to measure the edge and grade the pick.

**Pick win probability:** The estimated chance that the selected spread or total pick hits.

**Lock:** A pick that passes the model's win-probability, positive-value, sportsbook-support, and combined weekly-ranking rules.
