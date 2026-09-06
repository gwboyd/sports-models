# Expected Points 2.0 Draft

## Public Summary

Adjust CFB explosiveness features for opponent quality while accounting for the FBS/FCS divide.

## Changes

- Replace raw historical offense and defense explosiveness inputs with leakage-safe, ridge-regularized opponent-adjusted values under the existing feature names.
- Make the FCS pooling policy, ridge strength, and cross-season carryover configurable for backtesting.
- Preserve the established dynamic smoothing span based on the target game's week, including bye gaps and season rollover.

## Evaluation

Standard weekly replay for 2023–2025: 2,610 games and 47 cutoffs per recipe. Candidate `4394a37` versus deployed `1.0` at `4b3fa54`, evaluated September 6, 2026 with 2,000 fixed-seed season-week bootstrap samples. Both sides were fit without cache hits.

| Metric | Deployed | Candidate |
|---|---:|---:|
| Team-score MAE (points) | 9.1086 | 9.1061 |
| Margin MAE (points) | 12.6149 | 12.5943 |
| Total-points MAE (points) | 13.2449 | 13.2095 |
| Spread Brier score | 0.2754 | 0.2774 |
| Total Brier score | 0.2731 | 0.2725 |
| Spread-lock win rate (%) | 50.0000 | 48.7437 |
| Total-lock win rate (%) | 48.5437 | 54.2857 |

All pooled directional changes are statistically inconclusive at the 95% level. This evaluation does not establish an overall improvement or support promoting the current candidate unchanged. Historical feed replay is a weekly approximation, not an exact intraday reconstruction.
