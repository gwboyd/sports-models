"""Leakage-safe feature registries for the NFL/CFB lock probability heads."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ExpectedPointsLeague


FORBIDDEN_FEATURE_FRAGMENTS = (
    "home_score",
    "away_score",
    "postgame",
    "true_",
    "correct_",
    "_lock",
    "recipe_",
)
FORBIDDEN_FEATURE_COLUMNS = {
    "spread_win", "total_win", "recorded_probability", "p_win", "p_loss", "p_push",
}

CORE_NUMERIC = (
    "edge",
    "signed_edge",
    "spread_line",
    "total_line",
    "predicted_margin",
    "predicted_total",
    "line_is_integer",
    "spread_key_3_distance",
    "spread_key_7_distance",
)
CORE_CATEGORICAL = ("market", "direction")

NFL_CONTEXT_NUMERIC = (
    "picked_rest_advantage",
    "picked_qbr_advantage",
    "division_game",
    "implied_points_home",
    "implied_points_away",
    "ewma_dynamic_window_rushing_offense_home",
    "ewma_dynamic_window_rushing_offense_away",
    "ewma_dynamic_window_passing_offense_home",
    "ewma_dynamic_window_passing_offense_away",
    "ewma_dynamic_window_rushing_defense_home",
    "ewma_dynamic_window_rushing_defense_away",
    "ewma_dynamic_window_passing_defense_home",
    "ewma_dynamic_window_passing_defense_away",
    "ewma_success_rate_rushing_offense_home",
    "ewma_success_rate_rushing_offense_away",
    "ewma_success_rate_passing_offense_home",
    "ewma_success_rate_passing_offense_away",
    "ewma_success_rate_rushing_defense_home",
    "ewma_success_rate_rushing_defense_away",
    "ewma_success_rate_passing_defense_home",
    "ewma_success_rate_passing_defense_away",
)
NFL_CONTEXT_CATEGORICAL = ("roof", "weekday")
NFL_MARKET_NUMERIC = (
    "moneyline_home",
    "moneyline_away",
    "selected_spread_odds",
    "opposing_spread_odds",
    "over_odds_for_selected_play",
)

CFB_CONTEXT_NUMERIC = (
    "neutral_site",
    "conference_game",
    "picked_elo_advantage",
    "elo_gap",
    "implied_points_home",
    "implied_points_away",
    "offense_explosiveness_ewma_dynamic_window_home",
    "offense_explosiveness_ewma_dynamic_window_away",
    "defense_explosiveness_ewma_dynamic_window_home",
    "defense_explosiveness_ewma_dynamic_window_away",
    "offense_ppa_ewma_dynamic_window_home",
    "offense_ppa_ewma_dynamic_window_away",
    "defense_ppa_ewma_dynamic_window_home",
    "defense_ppa_ewma_dynamic_window_away",
    "offense_success_rate_ewma_dynamic_window_home",
    "offense_success_rate_ewma_dynamic_window_away",
    "defense_success_rate_ewma_dynamic_window_home",
    "defense_success_rate_ewma_dynamic_window_away",
)
CFB_CONTEXT_CATEGORICAL = (
    "weekday",
    "picked_conference",
    "opponent_conference",
    "home_classification",
    "away_classification",
)
CFB_MARKET_NUMERIC = (
    "reference_line",
    "open_line",
    "line_move",
    "line_move_abs",
    "quote_count",
    "quote_range",
    "shopping_points",
    "move_toward_pick",
    "consensus_home_win_prob",
)
CFB_MARKET_CATEGORICAL = ("provider", "selection_reason")


@dataclass(frozen=True)
class LockFeatureSet:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    promotion_eligible: bool

    @property
    def all(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.numeric, *self.categorical)))

    def validate(self) -> None:
        forbidden = [
            column
            for column in self.all
            if column in FORBIDDEN_FEATURE_COLUMNS
            or any(fragment in column for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
        ]
        if forbidden:
            raise ValueError(f"Lock feature registry contains forbidden columns: {forbidden}")


def lock_feature_set(
    league: ExpectedPointsLeague,
    group: str,
    *,
    market: str | None = None,
) -> LockFeatureSet:
    """Return an immutable allowlist; optional missing values are imputed, never discovered."""
    numeric = list(CORE_NUMERIC)
    categorical = list(CORE_CATEGORICAL)
    promotable = True
    if group in {"context", "market"}:
        if league is ExpectedPointsLeague.NFL:
            numeric.extend(NFL_CONTEXT_NUMERIC)
            categorical.extend(NFL_CONTEXT_CATEGORICAL)
        else:
            numeric.extend(CFB_CONTEXT_NUMERIC)
            categorical.extend(CFB_CONTEXT_CATEGORICAL)
    if group == "market":
        promotable = False
        if league is ExpectedPointsLeague.NFL:
            numeric.extend(
                column for column in NFL_MARKET_NUMERIC
                if market is None
                or (market == "spread" and column != "over_odds_for_selected_play")
                or (market == "total" and column not in {"selected_spread_odds", "opposing_spread_odds"})
            )
        else:
            numeric.extend(CFB_MARKET_NUMERIC)
            categorical.extend(CFB_MARKET_CATEGORICAL)
    elif group != "core" and group != "context":
        raise ValueError(f"Unknown lock feature group: {group}")
    result = LockFeatureSet(tuple(numeric), tuple(categorical), promotable)
    result.validate()
    return result


__all__ = ["LockFeatureSet", "lock_feature_set"]
