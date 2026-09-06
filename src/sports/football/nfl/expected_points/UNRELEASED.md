# Expected Points 2.0 Draft

## Public Summary

Adjust NFL team efficiency features for the quality of the offenses and defenses they faced.

## Changes

- Replace raw historical EPA and custom success-rate inputs with leakage-safe, ridge-regularized opponent-adjusted values under the existing feature names.
- Expose ridge strength and cross-season carryover as recipe settings for historical evaluation.
