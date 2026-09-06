# Expected Points 2.0 Draft

## Public Summary

Adjust CFB explosiveness features for opponent quality while accounting for the FBS/FCS divide.

## Changes

- Replace raw historical offense and defense explosiveness inputs with leakage-safe, ridge-regularized opponent-adjusted values under the existing feature names.
- Make the FCS pooling policy, ridge strength, and cross-season carryover configurable for backtesting.
