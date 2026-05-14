"""KPI engine: corridor-level KPIs + period-over-period (PoP) trend analysis.

Operates on ReconciliationResult objects produced by csv_tools. All numbers
returned here are precomputed and intended to be passed verbatim to the
report agent (the LLM must NOT recompute them).
"""
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List

from tools.dq_reconciler import ReconciledRow, ReconciliationResult


def _corridor_slice(rows: List[ReconciledRow], corridor_id: str) -> List[ReconciledRow]:
    return [r for r in rows if r.original.get("corridor_id") == corridor_id]


def compute_corridor_kpis(
    planning: ReconciliationResult,
    history: ReconciliationResult,
    corridor_ids: List[str],
) -> Dict[str, Any]:
    """Return per-corridor KPIs for the 48h planning window + PoP comparison.

    Per-corridor fields:
        - planning_valid_units, planning_excluded_units, planning_flagged_units
        - day0_units, day1_units
        - tier1_units, tier2_units                (from sla_tier on the row)
        - cold_chain_units, room_temp_units
        - top_destinations (Counter)
        - history_daily_avg, planning_daily_avg, pop_delta_pct
    """
    out: Dict[str, Any] = {}
    history_days = _count_unique_days(history.rows)
    planning_days = max(_count_unique_days(planning.rows), 1)  # at least 1

    for cid in corridor_ids:
        p_all = _corridor_slice(planning.rows, cid)
        p_valid = [r for r in p_all if r.is_valid_for_dispatch]
        p_excl  = [r for r in p_all if not r.is_valid_for_dispatch]

        day_split = Counter(r.original.get("planning_day") for r in p_valid)
        tier_split = Counter(r.sla_tier for r in p_valid)
        cold = sum(1 for r in p_valid if r.is_cold_chain)
        room = sum(1 for r in p_valid if r.is_cold_chain is False)
        dests = Counter(r.original.get("dispatch_location") for r in p_valid)

        h_valid = [r for r in history.valid_rows if r.original.get("corridor_id") == cid]
        h_days = max(_count_unique_days(h_valid), 1)
        history_daily_avg = round(len(h_valid) / h_days, 2)
        planning_daily_avg = round(len(p_valid) / planning_days, 2)
        pop_delta_pct = (
            round((planning_daily_avg - history_daily_avg) / history_daily_avg * 100.0, 1)
            if history_daily_avg > 0 else None
        )

        out[cid] = {
            "planning_valid_units":   len(p_valid),
            "planning_excluded_units": len(p_excl),
            "day0_units": day_split.get("Day0", 0),
            "day1_units": day_split.get("Day1", 0),
            "tier1_units": tier_split.get(1, 0),
            "tier2_units": tier_split.get(2, 0),
            "cold_chain_units": cold,
            "room_temp_units": room,
            "top_destinations": dict(dests.most_common(5)),
            "history_daily_avg_units": history_daily_avg,
            "planning_daily_avg_units": planning_daily_avg,
            "pop_delta_pct": pop_delta_pct,
            "history_window_days": h_days,
        }

    return out


def compute_overall_kpis(
    planning: ReconciliationResult,
    history: ReconciliationResult,
) -> Dict[str, Any]:
    return {
        "history": history.summary(),
        "planning": planning.summary(),
        "planning_cold_chain_units": sum(1 for r in planning.valid_rows if r.is_cold_chain),
        "planning_tier1_units": sum(1 for r in planning.valid_rows if r.sla_tier == 1),
        "planning_tier2_units": sum(1 for r in planning.valid_rows if r.sla_tier == 2),
    }


def _count_unique_days(rows: List[ReconciledRow]) -> int:
    dates = {r.original.get("shipment_date") for r in rows if r.original.get("shipment_date") is not None}
    return len({str(d) for d in dates})
