# Expected Points 2.0 Draft

## Public Summary

Adjust NFL team efficiency features for the quality of the offenses and defenses they faced.

## Changes

- Replace raw historical EPA and custom success-rate inputs with leakage-safe, ridge-regularized opponent-adjusted values under the existing feature names.
- Expose ridge strength and cross-season carryover as recipe settings for historical evaluation.
- Preserve the established dynamic smoothing span based on the target game's week, including bye gaps and season rollover.

## Evaluation

Candidate frames and comparison reports from before the target-week smoothing correction must be regenerated;
those earlier results do not evaluate the current draft.
