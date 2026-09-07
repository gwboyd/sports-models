"""Atomic, inspectable artifacts for lock-only replay studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from .backtest_artifacts import write_json_atomic, write_parquet_atomic, write_text_atomic
from .lock_replay import LockReplaySource, LockReplaySpec, LockSelection, registered_lock_variants
from .backtesting import recipe_code_identity


def write_lock_study_inputs(
    root: Path,
    source: LockReplaySource,
    spec: LockReplaySpec,
) -> None:
    write_json_atomic(root / "source.json", {
        "artifact_type": "expected_points_lock_replay_source",
        "source_path": str(source.path),
        "league": source.run.league.value,
        "recipe_name": source.run.recipe_name,
        "recipe_version": source.run.recipe_version,
        "recipe_fingerprint": source.run.recipe_fingerprint,
        "profile": source.run.profile,
        "seasons": source.run.seasons,
        "rows": len(source.run.predictions),
        "cutoffs": int(source.run.predictions["backtest_cutoff"].nunique()),
        "source_fingerprint": source.run.source_fingerprint,
        "prediction_fingerprint": source.prediction_fingerprint,
        "predictions_sha256": source.predictions_sha256,
        "historical_fidelity": (
            "Weekly pre-first-kickoff replay using currently available historical feeds; "
            "not an exact intraday line, public-money, or roster snapshot replay."
        ),
    })
    write_json_atomic(root / "design.json", {
        "artifact_type": "expected_points_lock_replay_design",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": asdict(spec),
        "variants": [asdict(variant) for variant in registered_lock_variants(source.run.league)],
        "code_identity": recipe_code_identity(source.run.league),
        "selection_rule": (
            "Development seasons only; >=0.1% relative Brier improvement versus shrunken base rate, "
            "non-worse log loss, no more than 5pp overconfidence, >=25 locks, and positive 95% "
            "season-week bootstrap lower bound for flat -110 net units."
        ),
    })
    write_parquet_atomic(root / "source_ledger.parquet", source.ledger)


def write_variant(
    root: Path,
    *,
    phase: str,
    market: str,
    name: str,
    decisions: pd.DataFrame,
    metrics: Mapping,
) -> Path:
    destination = root / phase / market / name
    destination.mkdir(parents=True, exist_ok=True)
    compact_columns = [
        "season", "week", "game_id", "market", "cutoff", "kickoff", "play", "line",
        "prediction", "edge", "recorded_probability", "recorded_lock", "variant",
        "feature_group", "family", "calibrator", "trained_rows", "trained_through",
        "resolved_win_probability", "p_win", "p_push", "p_loss", "expected_units",
        "market_supported", "locks_enabled", "lock_rank", "lock", "lock_reason", "outcome",
    ]
    write_parquet_atomic(destination / "decisions.parquet", decisions.loc[:, compact_columns])
    write_json_atomic(destination / "metrics.json", dict(metrics))
    return destination


def write_selection(root: Path, selection: LockSelection) -> Path:
    destination = root / "selection.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("selection_fingerprint") != selection.selection_fingerprint:
            raise ValueError("Refusing to overwrite a different frozen lock selection")
        return destination
    write_json_atomic(destination, asdict(selection))
    return destination


def load_selection(root: Path) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    fingerprint = selection.pop("selection_fingerprint")
    expected = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected:
        raise ValueError("Frozen lock selection fingerprint does not match its contents")
    selection["selection_fingerprint"] = fingerprint
    return selection


def render_lock_study_report(
    source: LockReplaySource,
    selection: LockSelection,
    confirmation: Mapping[str, Mapping] | None = None,
    promotion: Mapping | None = None,
) -> str:
    lines = [
        "# Expected-Points Lock Replay Study",
        "",
        f"League: {source.run.league.value.upper()}",
        "",
        f"Frozen source: `{source.run.recipe_version}`; {len(source.run.predictions):,} games",
        "",
        "> Results are flat-stake simulations at -110 from weekly historical feed snapshots, not exact intraday execution.",
        "",
        "## Frozen selection",
        "",
        f"- Spread probability head: `{selection.spread_variant}`",
        f"- Total probability head: `{selection.total_variant}`",
        f"- Lock-enabled markets: {', '.join(selection.enabled_markets) or 'none'}",
    ]
    if selection.rejection_reasons:
        lines.extend(["", "Selection notes:", ""])
        lines.extend(f"- {reason}" for reason in selection.rejection_reasons)
    if confirmation:
        lines.extend(["", "## Confirmation", "", "| Market | Locks | Record | Units | ROI | Brier | Log loss |", "|---|---:|---:|---:|---:|---:|---:|"])
        for market, metrics in confirmation.items():
            roi = "—" if metrics.get("roi") is None else f"{100 * metrics['roi']:.1f}%"
            lines.append(
                f"| {market} | {metrics.get('locks', 0)} | {metrics.get('wins', 0)}-"
                f"{metrics.get('losses', 0)}-{metrics.get('pushes', 0)} | "
                f"{metrics.get('net_units', 0):.2f} | {roi} | "
                f"{metrics.get('brier') if metrics.get('brier') is not None else '—'} | "
                f"{metrics.get('log_loss') if metrics.get('log_loss') is not None else '—'} |"
            )
    if promotion:
        lines.extend([
            "", "## Production promotion", "",
            f"Promoted markets: {', '.join(promotion.get('promoted_markets', [])) or 'none'}", "",
        ])
        for market, result in promotion.get("markets", {}).items():
            failed = [name for name, passed in result["checks"].items() if not passed]
            lines.append(
                f"- {market}: {'passed' if result['promoted'] else 'not promoted'}"
                + (f"; failed {', '.join(failed)}" if failed else "")
            )
    return "\n".join(lines) + "\n"


def write_report(
    root: Path,
    source: LockReplaySource,
    selection: LockSelection,
    confirmation: Mapping[str, Mapping] | None = None,
    promotion: Mapping | None = None,
) -> None:
    write_text_atomic(
        root / "report.md",
        render_lock_study_report(source, selection, confirmation, promotion),
    )


__all__ = [
    "load_selection",
    "render_lock_study_report",
    "write_lock_study_inputs",
    "write_report",
    "write_selection",
    "write_variant",
]
