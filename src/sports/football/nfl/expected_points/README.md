# NFL Expected Points Model

The model predicts **expected points scored** for each NFL team in a game, getting to a predicted score from which a spread and total can be derived. We then compare those spreads and totals to the Vegas prices to make the picks each week. 

Metrics derived from play by play data is used starting from 2010 to now. EPA (expected points added) Per Play and Success Rate are the features the model has found the most useful to predict outcomes, but other features like staring quarterback QBR, days of rest, and the Vegas odds themselves are used.

After picks are made, there is another model (classifier) that looks back on the historical picks the model has made against Vegas, analyzes patterns with which the mdoel has been succesful, and gives a percentage chance it belives the model has of being correct in it's pick. That score (along with a couple other heuristics) is how we decide what "plays" to make each week.

## Picks Update Cadence

The model can be updated whenever, but is scheduled to and should always update at the following times:
- Wendesday at 12:00 AM EST
- Thursday at 7:20 PM EST
- Sunday at 12:00 PM EST

The model is "not allowed" to make picks within 30 mins of kickoff, so it's final prediction will be locked in at that time.

I do update results in batch manually at inconsistent times because I like to review how the model did each given week. I may eventually schedule this as well though.

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

The training dataset for these classifiers is the test set of the 80/20 test/train split we make to get a sense of accuracy. We can't use predictions of games the model has been trained on, since the model would have seen it before. Therefore, we must train the classifier on the test set, setting up a bit of a tradeoff between expected points data volume (at least at time of classifier training) and classification model volume depending on the split.

**A common question I get asked is "Why is the model not confident in the pick even though the predicted spread is so far off the Vegas spread?"** The answer lies in that the "confidence score" comes from this objective 3rd party model, and in a case of high diofference of spreads/totals and low confience, is saying it has seen similar scenarios before where the model has lost and it is therefore not that confident in the pick.


## Misc

### Power Rankings

The power rankings seen above in the chart are created by taking all of the metrics (so not inclusing odds, enviroment, rest, etc) for teams, training a classification model that predicts winning, and them creating simulations to where each team plays every other team home and away. The model outputs a win probability for esach game, so the teams "win percentage" is just and average of the win probabilities for all 62 (31 * 2) games a team would play.

The game simulations mimick each team's form for the current week (the next week if a team is on a bye), so it would be as if they all played eachother "today."

### Ideas for the future

- Add in more player specific data besides just quarterbacks to better account for injuries/trades
- Adjust efficiency metrics for difficulty of opponent
- Include explosiveness data 
- Add some 3rd down specifc metrics data
- Include some position group specific data to better catch matchup advantages
- Revamp how metrics are handled for rookies (only player specific metric right now is QBR)
- Get to a place where I can remove any information that relies on Vegas odds.. it helps but sorta feels like cheating
- If I do keep Vegas lines, find a way to get opening odds or where the public is so I better arbitrage the "vibes vs metrics" dynamic
- College football model for the 2025 season

