"""Deterministic resource allocator.

Greedy penalty-minimizing dispatch planner. Operates on:
    - planning-window valid shipment units (post-reconciliation)
    - 48h resource pools (drivers / truck_standard / truck_temp_controlled per day)
    - playbook penalty model + corridor SLA tiers
    - weather-derived per-corridor SLA-violation flag (from compute_sla_violation_from_weather)

Allocation strategy
-------------------
1. Compute base priority per unit:
       priority = (tier1?100 : tier2?40) + (cold_chain?80 : 0)
   Higher = served first.
2. Sort all planning-window valid units by priority DESC, then by
   (requested_day asc) so Day0 wins ties.
3. For each unit, try to place it on its requested day. If the day is
   capacity-blocked, try the other day (incurs +10 day-delay penalty).
   If neither day has capacity, the unit is unserved (incurs full SLA
   violation + cold-chain bonus).
4. Capacity model per day:
       trucks_used_standard  = ceil(units_assigned_standard / 9)
       trucks_used_temp_ctrl = ceil(units_assigned_cold     / 9)
       drivers_used          = trucks_used_standard + trucks_used_temp_ctrl
   Each must be <= the day's available pool.
5. After greedy assignment, layer in weather-driven SLA-violation penalties:
   if a corridor's weather buffer pushes transit time above its SLA cap,
   every served unit on that corridor incurs the corresponding tier SLA
   penalty (+cold-chain bonus if cold-chain).

The output is structured for the report agent to cite directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from math import ceil
from typing import Any, Dict, List, Optional

from tools.dq_reconciler import ReconciledRow


PLANNING_DAYS = ("Day0", "Day1")


@dataclass
class UnitAssignment:
    unit_uid: str
    corridor_id: str
    canonical_item_id: str
    sla_tier: int
    is_cold_chain: bool
    requested_day: str
    assigned_day: Optional[str]
    truck_type: Optional[str]
    served: bool
    penalty_breakdown: Dict[str, int] = field(default_factory=dict)
    penalty_total: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DayResourceUsage:
    day: str
    drivers_available: int
    truck_standard_available: int
    truck_temp_controlled_available: int
    units_standard: int = 0
    units_cold_chain: int = 0
    trucks_standard_used: int = 0
    trucks_temp_used: int = 0
    drivers_used: int = 0

    def can_fit(self, is_cold_chain: bool, effective_units_per_truck: int) -> bool:
        """Check if we can place ONE more unit of the given type on this day."""
        if is_cold_chain:
            new_cold = self.units_cold_chain + 1
            new_temp_trucks = ceil(new_cold / effective_units_per_truck)
            new_drivers = self.trucks_standard_used + new_temp_trucks
            return (new_temp_trucks <= self.truck_temp_controlled_available
                    and new_drivers <= self.drivers_available)
        else:
            new_std = self.units_standard + 1
            new_std_trucks = ceil(new_std / effective_units_per_truck)
            new_drivers = new_std_trucks + self.trucks_temp_used
            return (new_std_trucks <= self.truck_standard_available
                    and new_drivers <= self.drivers_available)

    def place(self, is_cold_chain: bool, effective_units_per_truck: int) -> None:
        if is_cold_chain:
            self.units_cold_chain += 1
            self.trucks_temp_used = ceil(self.units_cold_chain / effective_units_per_truck)
        else:
            self.units_standard += 1
            self.trucks_standard_used = ceil(self.units_standard / effective_units_per_truck)
        self.drivers_used = self.trucks_standard_used + self.trucks_temp_used


@dataclass
class AllocationResult:
    assignments: List[UnitAssignment]
    usage_by_day: Dict[str, DayResourceUsage]
    weather_sla_violations_by_corridor: Dict[str, bool]
    total_penalty: int
    penalty_breakdown: Dict[str, int]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments":   [a.to_dict() for a in self.assignments],
            "usage_by_day":  {d: asdict(u) for d, u in self.usage_by_day.items()},
            "weather_sla_violations_by_corridor": self.weather_sla_violations_by_corridor,
            "total_penalty": self.total_penalty,
            "penalty_breakdown": self.penalty_breakdown,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------

def _unit_base_priority(tier: int, is_cold_chain: bool, penalty_model: Dict[str, int]) -> int:
    sla = penalty_model["tier1_sla_violation"] if tier == 1 else penalty_model["tier2_sla_violation"]
    bonus = penalty_model["cold_chain_violation_bonus"] if is_cold_chain else 0
    return sla + bonus


def allocate(
    planning_valid_rows: List[ReconciledRow],
    resources_by_day: Dict[str, Dict[str, Any]],
    penalty_model: Dict[str, int],
    effective_units_per_truck: int,
    weather_report: Dict[str, Any],
    corridors: Dict[str, Any],
) -> AllocationResult:
    # Step 1: which corridors have a weather-driven SLA violation?
    weather_sla = _compute_weather_sla_violations(weather_report, corridors)

    # Step 2: initialize per-day resource usage trackers
    usage: Dict[str, DayResourceUsage] = {}
    for day in PLANNING_DAYS:
        r = resources_by_day.get(day, {})
        usage[day] = DayResourceUsage(
            day=day,
            drivers_available=int(r.get("driver", 0)),
            truck_standard_available=int(r.get("truck_standard", 0)),
            truck_temp_controlled_available=int(r.get("truck_temp_controlled", 0)),
        )

    # Step 3: sort units by priority desc, then requested_day asc, then corridor for stability
    enriched = []
    for row in planning_valid_rows:
        tier = row.sla_tier or 2
        is_cold = bool(row.is_cold_chain)
        priority = _unit_base_priority(tier, is_cold, penalty_model)
        enriched.append((priority, row, tier, is_cold))
    enriched.sort(
        key=lambda x: (
            -x[0],
            str(x[1].original.get("planning_day") or "Day9"),
            str(x[1].original.get("corridor_id") or ""),
            str(x[1].original.get("unique_item_id") or ""),
        ),
    )

    assignments: List[UnitAssignment] = []

    # Step 4: place each unit
    for priority, row, tier, is_cold in enriched:
        requested = row.original.get("planning_day")
        corridor = row.original.get("corridor_id")
        uid = str(row.original.get("unique_item_id"))
        canonical = row.canonical_item_id or "(unresolved)"

        # Try requested day first
        target_day: Optional[str] = None
        truck_type: Optional[str] = None
        delay_penalty = 0
        if requested in PLANNING_DAYS and usage[requested].can_fit(is_cold, effective_units_per_truck):
            target_day = requested
        else:
            # Try the other day
            alt = "Day1" if requested == "Day0" else "Day0"
            if alt in PLANNING_DAYS and usage[alt].can_fit(is_cold, effective_units_per_truck):
                target_day = alt
                delay_penalty = penalty_model["non_sla_day_delay"]

        breakdown: Dict[str, int] = {}
        served = False
        reason = ""

        if target_day is not None:
            usage[target_day].place(is_cold, effective_units_per_truck)
            truck_type = "truck_temp_controlled" if is_cold else "truck_standard"
            served = True
            if delay_penalty > 0:
                breakdown["non_sla_day_delay"] = delay_penalty
                reason = f"Bumped from {requested} to {target_day} (capacity)."
            else:
                reason = f"Dispatched on requested {target_day}."

            # Weather-driven SLA violation applies to served units too
            if weather_sla.get(corridor, False):
                if tier == 1:
                    breakdown["tier1_sla_violation_weather"] = penalty_model["tier1_sla_violation"]
                else:
                    breakdown["tier2_sla_violation_weather"] = penalty_model["tier2_sla_violation"]
                if is_cold:
                    breakdown["cold_chain_violation_bonus_weather"] = penalty_model["cold_chain_violation_bonus"]
                reason += " | Weather buffer breaches SLA cap on this corridor."
        else:
            # Unserved -> full SLA violation
            if tier == 1:
                breakdown["tier1_sla_violation_unserved"] = penalty_model["tier1_sla_violation"]
            else:
                breakdown["tier2_sla_violation_unserved"] = penalty_model["tier2_sla_violation"]
            if is_cold:
                breakdown["cold_chain_violation_bonus_unserved"] = penalty_model["cold_chain_violation_bonus"]
            reason = "Unserved: no truck capacity on either day."

        total = sum(breakdown.values())
        assignments.append(UnitAssignment(
            unit_uid=uid,
            corridor_id=corridor or "",
            canonical_item_id=canonical,
            sla_tier=tier,
            is_cold_chain=is_cold,
            requested_day=requested or "",
            assigned_day=target_day,
            truck_type=truck_type,
            served=served,
            penalty_breakdown=breakdown,
            penalty_total=total,
            reason=reason,
        ))

    # Step 5: aggregate
    total_penalty = sum(a.penalty_total for a in assignments)
    pb_total: Dict[str, int] = {}
    for a in assignments:
        for k, v in a.penalty_breakdown.items():
            pb_total[k] = pb_total.get(k, 0) + v

    served_count   = sum(1 for a in assignments if a.served)
    unserved_count = len(assignments) - served_count
    tier1_unserved = sum(1 for a in assignments if not a.served and a.sla_tier == 1)
    cold_unserved  = sum(1 for a in assignments if not a.served and a.is_cold_chain)
    bumped_count   = sum(1 for a in assignments if a.served and a.assigned_day != a.requested_day)

    summary = {
        "units_total": len(assignments),
        "units_served": served_count,
        "units_unserved": unserved_count,
        "units_bumped": bumped_count,
        "tier1_unserved": tier1_unserved,
        "cold_chain_unserved": cold_unserved,
        "weather_sla_violated_corridors": [c for c, v in weather_sla.items() if v],
    }

    return AllocationResult(
        assignments=assignments,
        usage_by_day=usage,
        weather_sla_violations_by_corridor=weather_sla,
        total_penalty=total_penalty,
        penalty_breakdown=pb_total,
        summary=summary,
    )


def _compute_weather_sla_violations(
    weather_report: Dict[str, Any],
    corridors: Dict[str, Any],
) -> Dict[str, bool]:
    """For each corridor, check if base_transit * (1 + buffer/100) > max_transit."""
    out: Dict[str, bool] = {}
    for cid, w in weather_report.items():
        cfg = corridors.get(cid, {})
        base = float(cfg.get("base_transit_hours", 0.0))
        cap  = float(cfg.get("max_transit_hours", 9e9))
        adj  = base * (1 + float(w.get("buffer_pct", 0)) / 100.0)
        out[cid] = adj > cap
    return out
