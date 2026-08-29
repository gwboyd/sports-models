"""Deterministic CFBD market consensus and execution-line selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


CLUSTER_MAX_RANGE = 2.0
SYNTHETIC_PROVIDERS = frozenset({"consensus", "numberfire", "teamrankings"})
PROVIDER_ALIASES = {
    "bovada": "Bovada",
    "caesars": "Caesars",
    "caesars sportsbook": "Caesars",
    "draft kings": "DraftKings",
    "draftkings": "DraftKings",
    "espn bet": "ESPN Bet",
    "william hill": "William Hill (New Jersey)",
    "william hill nj": "William Hill (New Jersey)",
    "william hill (new jersey)": "William Hill (New Jersey)",
}
PROVIDER_PRIORITY = (
    "Bovada",
    "William Hill (New Jersey)",
    "ESPN Bet",
    "DraftKings",
)

MARKET_DIAGNOSTIC_COLUMNS = (
    "spread_provider",
    "total_provider",
    "spread_reference_line",
    "total_reference_line",
    "spread_open_line",
    "total_open_line",
    "spread_quote_count",
    "total_quote_count",
    "spread_quote_range",
    "total_quote_range",
    "spread_market_supported",
    "total_market_supported",
    "spread_selection_reason",
    "total_selection_reason",
    "spread_shopping_points",
    "total_shopping_points",
)

MARKET_COLUMNS = (
    "id",
    "spread_reference_line",
    "total_reference_line",
    "spread_open_line",
    "total_open_line",
    "spread_line_move",
    "total_line_move",
    "spread_line_move_abs",
    "total_line_move_abs",
    "spread_quote_count",
    "total_quote_count",
    "spread_quote_range",
    "total_quote_range",
    "spread_market_supported",
    "total_market_supported",
    "spread_selection_reason",
    "total_selection_reason",
    "spread_execution_candidates",
    "total_execution_candidates",
    "consensus_home_win_prob",
    "consensus_home_win_prob_missing",
    "implied_points_home",
    "implied_points_away",
)

OPTIONAL_CONFIDENCE_FEATURE_GROUPS = {
    "spread": {
        "opening": ("spread_open_line",),
        "movement": (
            "spread_line_move",
            "spread_line_move_abs",
            "spread_move_toward_pick",
        ),
        "market_depth": (
            "spread_quote_count",
            "spread_quote_range",
            "spread_market_supported",
            "consensus_home_win_prob",
            "consensus_home_win_prob_missing",
        ),
    },
    "total": {
        "opening": ("total_open_line",),
        "movement": (
            "total_line_move",
            "total_line_move_abs",
            "total_move_toward_pick",
        ),
        "market_depth": (
            "total_quote_count",
            "total_quote_range",
            "total_market_supported",
            "consensus_home_win_prob",
            "consensus_home_win_prob_missing",
        ),
    },
}


@dataclass(frozen=True)
class ProviderQuote:
    provider: str
    value: float


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _median(values: Iterable[float]) -> float | None:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    return float(np.median(finite)) if finite else None


def _provider_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    provider = " ".join(value.split())
    provider_key = provider.casefold()
    if provider_key in SYNTHETIC_PROVIDERS:
        return None
    return PROVIDER_ALIASES.get(provider_key, provider)


def _provider_rank(provider: str) -> tuple[int, str]:
    if provider in PROVIDER_PRIORITY:
        return PROVIDER_PRIORITY.index(provider), provider
    if provider.casefold().startswith("caesars"):
        return len(PROVIDER_PRIORITY), provider
    return len(PROVIDER_PRIORITY) + 1, provider


def _quote_value(quote: Mapping[str, Any], camel: str, snake: str) -> Any:
    return quote.get(camel, quote.get(snake))


def _collapse_provider_quotes(raw_quotes: Any) -> dict[str, dict[str, float | None]]:
    values: dict[str, dict[str, list[float]]] = {}
    if not isinstance(raw_quotes, Sequence) or isinstance(raw_quotes, (str, bytes)):
        return {}

    fields = {
        "spread": ("spread", "spread"),
        "spread_open": ("spreadOpen", "spread_open"),
        "total": ("overUnder", "over_under"),
        "total_open": ("overUnderOpen", "over_under_open"),
        "home_moneyline": ("homeMoneyline", "home_moneyline"),
        "away_moneyline": ("awayMoneyline", "away_moneyline"),
    }
    for raw_quote in raw_quotes:
        if not isinstance(raw_quote, Mapping):
            continue
        provider = _provider_name(raw_quote.get("provider"))
        if provider is None:
            continue
        provider_values = values.setdefault(provider, {field: [] for field in fields})
        for field, aliases in fields.items():
            parsed = _finite_float(_quote_value(raw_quote, *aliases))
            if parsed is not None:
                provider_values[field].append(parsed)

    return {
        provider: {field: _median(field_values) for field, field_values in provider_values.items()}
        for provider, provider_values in values.items()
    }


def _cluster_sort_key(
    cluster: tuple[ProviderQuote, ...],
    reference: float,
) -> tuple[Any, ...]:
    cluster_median = float(np.median([quote.value for quote in cluster]))
    return (
        -len(cluster),
        abs(cluster_median - reference),
        tuple(sorted(_provider_rank(quote.provider) for quote in cluster)),
        tuple((quote.value, quote.provider) for quote in cluster),
    )


def _largest_cluster(
    quotes: tuple[ProviderQuote, ...],
    reference: float,
    max_range: float,
) -> tuple[ProviderQuote, ...]:
    clusters: list[tuple[ProviderQuote, ...]] = []
    for start in range(len(quotes)):
        end = start
        while end + 1 < len(quotes) and quotes[end + 1].value - quotes[start].value <= max_range:
            end += 1
        clusters.append(quotes[start : end + 1])
    return min(clusters, key=lambda cluster: _cluster_sort_key(cluster, reference))


def _market_summary(
    provider_quotes: Mapping[str, Mapping[str, float | None]],
    *,
    current_field: str,
    opening_field: str,
    max_cluster_range: float,
) -> dict[str, Any] | None:
    quotes = tuple(
        sorted(
            (
                ProviderQuote(provider, float(value))
                for provider, fields in provider_quotes.items()
                if (value := fields.get(current_field)) is not None
            ),
            key=lambda quote: (quote.value, _provider_rank(quote.provider)),
        )
    )
    if not quotes:
        return None

    reference = float(np.median([quote.value for quote in quotes]))
    cluster = _largest_cluster(quotes, reference, max_cluster_range)
    supported = len(cluster) >= 2
    if supported:
        candidates = cluster
        reason = "corroborated_cluster"
    else:
        candidates = (
            min(
                quotes,
                key=lambda quote: (
                    abs(quote.value - reference),
                    _provider_rank(quote.provider),
                    quote.value,
                ),
            ),
        )
        reason = "single_quote" if len(quotes) == 1 else "uncorroborated_fallback"

    opening = _median(
        fields.get(opening_field)
        for fields in provider_quotes.values()
        if fields.get(opening_field) is not None
    )
    values = [quote.value for quote in quotes]
    return {
        "reference_line": reference,
        "opening_line": opening,
        "quote_count": len(quotes),
        "quote_range": float(max(values) - min(values)),
        "market_supported": supported,
        "selection_reason": reason,
        "execution_candidates": tuple((quote.provider, quote.value) for quote in candidates),
    }


def _american_implied_probability(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _consensus_no_vig_home_probability(
    provider_quotes: Mapping[str, Mapping[str, float | None]],
) -> float | None:
    probabilities: list[float] = []
    for fields in provider_quotes.values():
        home = _american_implied_probability(fields.get("home_moneyline"))
        away = _american_implied_probability(fields.get("away_moneyline"))
        if home is None or away is None or home + away <= 0:
            continue
        probabilities.append(home / (home + away))
    return _median(probabilities)


def assemble_cfb_betting_markets(
    lines: pd.DataFrame,
    *,
    max_cluster_range: float = CLUSTER_MAX_RANGE,
) -> pd.DataFrame:
    """Build one deterministic, model-ready market row per FBS-involved game."""
    required = {"id", "lines", "home_classification", "away_classification"}
    missing = sorted(required - set(lines.columns))
    if missing:
        raise ValueError(f"CFBD lines frame is missing required columns: {', '.join(missing)}")
    if max_cluster_range < 0:
        raise ValueError("max_cluster_range must be non-negative")

    rows: list[dict[str, Any]] = []
    for row in lines.to_dict(orient="records"):
        classifications = {
            str(row.get("home_classification", "")).strip().casefold(),
            str(row.get("away_classification", "")).strip().casefold(),
        }
        if "fbs" not in classifications:
            continue

        provider_quotes = _collapse_provider_quotes(row.get("lines"))
        spread = _market_summary(
            provider_quotes,
            current_field="spread",
            opening_field="spread_open",
            max_cluster_range=max_cluster_range,
        )
        total = _market_summary(
            provider_quotes,
            current_field="total",
            opening_field="total_open",
            max_cluster_range=max_cluster_range,
        )
        if spread is None or total is None:
            continue

        no_vig_home = _consensus_no_vig_home_probability(provider_quotes)
        spread_reference = spread["reference_line"]
        total_reference = total["reference_line"]
        rows.append(
            {
                "id": row["id"],
                "spread_reference_line": spread_reference,
                "total_reference_line": total_reference,
                "spread_open_line": spread["opening_line"],
                "total_open_line": total["opening_line"],
                "spread_line_move": (
                    spread_reference - spread["opening_line"]
                    if spread["opening_line"] is not None
                    else np.nan
                ),
                "total_line_move": (
                    total_reference - total["opening_line"]
                    if total["opening_line"] is not None
                    else np.nan
                ),
                "spread_line_move_abs": (
                    abs(spread_reference - spread["opening_line"])
                    if spread["opening_line"] is not None
                    else np.nan
                ),
                "total_line_move_abs": (
                    abs(total_reference - total["opening_line"])
                    if total["opening_line"] is not None
                    else np.nan
                ),
                "spread_quote_count": spread["quote_count"],
                "total_quote_count": total["quote_count"],
                "spread_quote_range": spread["quote_range"],
                "total_quote_range": total["quote_range"],
                "spread_market_supported": spread["market_supported"],
                "total_market_supported": total["market_supported"],
                "spread_selection_reason": spread["selection_reason"],
                "total_selection_reason": total["selection_reason"],
                "spread_execution_candidates": spread["execution_candidates"],
                "total_execution_candidates": total["execution_candidates"],
                "consensus_home_win_prob": no_vig_home if no_vig_home is not None else np.nan,
                "consensus_home_win_prob_missing": int(no_vig_home is None),
                "implied_points_home": (total_reference / 2.0) - (spread_reference / 2.0),
                "implied_points_away": (total_reference / 2.0) + (spread_reference / 2.0),
            }
        )
    return pd.DataFrame.from_records(rows, columns=MARKET_COLUMNS)


def _select_execution_quote(
    raw_candidates: Any,
    *,
    maximize: bool,
) -> ProviderQuote:
    if not isinstance(raw_candidates, Sequence) or not raw_candidates:
        raise ValueError("Market execution candidates cannot be empty")
    candidates = tuple(ProviderQuote(str(provider), float(value)) for provider, value in raw_candidates)
    return min(
        candidates,
        key=lambda quote: (
            -quote.value if maximize else quote.value,
            _provider_rank(quote.provider),
        ),
    )


def scores_to_cfb_bets(results: pd.DataFrame) -> pd.DataFrame:
    """Choose direction against consensus, then shop a credible execution quote."""
    required = {
        "home_team",
        "away_team",
        "home_score_pred",
        "away_score_pred",
        "spread_reference_line",
        "total_reference_line",
        "spread_execution_candidates",
        "total_execution_candidates",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"CFB betting frame is missing required columns: {', '.join(missing)}")

    output = results.copy()
    output["spread_pred"] = output["away_score_pred"] - output["home_score_pred"]
    output["total_pred"] = output["home_score_pred"] + output["away_score_pred"]
    output["spread_play"] = output.apply(
        lambda row: row["home_team"]
        if row["spread_pred"] < row["spread_reference_line"]
        else row["away_team"],
        axis=1,
    )
    output["total_play"] = output.apply(
        lambda row: "under"
        if row["total_pred"] < row["total_reference_line"]
        else ("over" if row["total_pred"] > row["total_reference_line"] else None),
        axis=1,
    )

    spread_quotes = output.apply(
        lambda row: _select_execution_quote(
            row["spread_execution_candidates"],
            maximize=row["spread_play"] == row["home_team"],
        ),
        axis=1,
    )
    total_quotes = output.apply(
        lambda row: _select_execution_quote(
            row["total_execution_candidates"],
            maximize=row["total_play"] == "under",
        ),
        axis=1,
    )
    output["spread_provider"] = spread_quotes.map(lambda quote: quote.provider)
    output["spread_line"] = spread_quotes.map(lambda quote: quote.value)
    output["total_provider"] = total_quotes.map(lambda quote: quote.provider)
    output["total_line"] = total_quotes.map(lambda quote: quote.value)
    output["spread_diff"] = (output["spread_line"] - output["spread_pred"]).abs()
    output["total_diff"] = (output["total_line"] - output["total_pred"]).abs()

    output["spread_shopping_points"] = output.apply(
        lambda row: (
            row["spread_line"] - row["spread_reference_line"]
            if row["spread_play"] == row["home_team"]
            else row["spread_reference_line"] - row["spread_line"]
        ),
        axis=1,
    )
    output["total_shopping_points"] = output.apply(
        lambda row: (
            row["total_reference_line"] - row["total_line"]
            if row["total_play"] == "over"
            else row["total_line"] - row["total_reference_line"]
        ),
        axis=1,
    )
    output["spread_move_toward_pick"] = output.apply(
        lambda row: (
            -row["spread_line_move"]
            if row["spread_play"] == row["home_team"]
            else row["spread_line_move"]
        ),
        axis=1,
    )
    output["total_move_toward_pick"] = output.apply(
        lambda row: (
            row["total_line_move"]
            if row["total_play"] == "over"
            else -row["total_line_move"]
        ),
        axis=1,
    )
    return output


__all__ = [
    "CLUSTER_MAX_RANGE",
    "MARKET_DIAGNOSTIC_COLUMNS",
    "OPTIONAL_CONFIDENCE_FEATURE_GROUPS",
    "assemble_cfb_betting_markets",
    "scores_to_cfb_bets",
]
