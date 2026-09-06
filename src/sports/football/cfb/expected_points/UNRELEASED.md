# Expected Points 2.0 Draft

## Public Summary

Adjust CFB explosiveness features for opponent quality while accounting for the FBS/FCS divide.

## Changes

- Replace raw historical offense and defense explosiveness inputs with leakage-safe, ridge-regularized opponent-adjusted values under the existing feature names.
- Make the FCS pooling policy, ridge strength, and cross-season carryover configurable for backtesting.
- Preserve the established dynamic smoothing span based on the target game's week, including bye gaps and season rollover.

## Evaluation

Candidate frames and comparison reports from before the target-week smoothing correction must be regenerated;
those earlier results do not evaluate the current draft.
