# Expected Points 2.0 Draft

## Public Summary

Adjust NFL team efficiency features for the quality of the offenses and defenses they faced.

## Changes

- Replace raw historical EPA and custom success-rate inputs with leakage-safe, ridge-regularized opponent-adjusted values under the existing feature names.
- Expose ridge strength and cross-season carryover as recipe settings for historical evaluation.
- Preserve the established dynamic smoothing span based on the target game's week, including bye gaps and season rollover.

## Evaluation

Standard weekly replay for 2023–2025: 855 games and 66 cutoffs per recipe. Candidate `4394a37` versus deployed `1.0` at `4b3fa54`, evaluated September 6, 2026 with 2,000 fixed-seed season-week bootstrap samples. Both sides were fit without cache hits.

| Metric | Deployed | Candidate |
|---|---:|---:|
| Team-score MAE (points) | 7.4347 | 7.4023 |
| Margin MAE (points) | 10.2084 | 10.1015 |
| Total-points MAE (points) | 10.4511 | 10.3305 |
| Spread Brier score | 0.2933 | 0.3013 |
| Total Brier score | 0.2934 | 0.2980 |
| Spread-lock win rate (%) | 47.0085 | 42.9245 |
| Total-lock win rate (%) | 45.5285 | 50.4673 |

All pooled directional changes are statistically inconclusive at the 95% level. This evaluation does not establish an overall improvement or support promoting the current candidate unchanged. Historical feed replay is a weekly approximation, not an exact intraday reconstruction.
