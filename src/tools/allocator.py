"""Deterministic resource allocator (ILP).

Solves the SeeWeeS playbook §13 penalty model as a mixed-integer linear
program using PuLP + the bundled CBC solver. Produces a provably-optimal
dispatch plan in <1 second for the planning-window scale (~33 units).

INPUTS
    - planning-window valid shipment units (post-reconciliation)
    - 48h resource pools (drivers / truck_standard / truck_temp_controlled per day)
    - playbook penalty model + corridor SLA tiers
    - per-corridor weather report (drives the weather-induced SLA flag)

DECISION VARIABLES
    x[u, d] in {0, 1}     unit u placed on day d (Day0 or Day1)
    tc[d] integer >= 0    temp-controlled trucks used on day d
    ts[d] integer >= 0    standard trucks used on day d

CONSTRAINTS
    Each unit placed at most once:        sum_d x[u, d] <= 1
    Cold-chain unit volume per day:       sum_{u cold} x[u, d] <= EFFECTIVE * tc[d]
    Standard unit volume per day:         sum_{u std}  x[u, d] <= EFFECTIVE * ts[d]
    Truck pools:                          tc[d] <= reefer_available[d]
                                          ts[d] <= standard_available[d]
    Driver pool (one driver per truck):   tc[d] + ts[d] <= drivers_available[d]

OBJECTIVE  (minimize)
    For each unit:
        + (1 - sum_d x[u, d]) * full_unserved_penalty(u)
        + day_delay_penalty * 1[bumped from requested day to alternate day]
        + weather_sla_penalty(u) * sum_d x[u, d]   if unit's corridor has a
                                                   weather-induced SLA breach
    where full_unserved_penalty(u) = tier_sla_violation + (cold? cold_chain_bonus : 0)
    and   day_delay_penalty       = playbook 'non_sla_day_delay' (=10)

The output shape (AllocationResult) is identical to the previous greedy
allocator so downstream code, tests, and the report payload don't change.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from math import ceil
from typing import Any, Dict, List, Optional

import pulp

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


@dataclass
class AllocationResult:
    assignments: List[UnitAssignment]
    usage_by_day: Dict[str, DayResourceUsage]
    weather_sla_violations_by_corridor: Dict[str, bool]
    total_penalty: int
    penalty_breakdown: Dict[str, int]
    summary: Dict[str, Any]
    solver_status: str = "optimal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments":   [a.to_dict() for a in self.assignments],
            "usage_by_day":  {d: asdict(u) for d, u in self.usage_by_day.items()},
            "weather_sla_violations_by_corridor": self.weather_sla_violations_by_corridor,
            "total_penalty": self.total_penalty,
            "penalty_breakdown": self.penalty_breakdown,
            "summary": self.summary,
            "solver_status": self.solver_status,
        }


# ---------------------------------------------------------------------------

def _compute_weather_sla_violations(
    weather_report: Dict[str, Any],
    corridors: Dict[str, Any],
) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for cid, w in weather_report.items():
        cfg = corridors.get(cid, {})
        base = float(cfg.get("base_transit_hours", 0.0))
        cap  = float(cfg.get("max_transit_hours", 9e9))
        adj  = base * (1 + float(w.get("buffer_pct", 0)) / 100.0)
        out[cid] = adj > cap
    return out


def allocate(
    planning_valid_rows: List[ReconciledRow],
    resources_by_day: Dict[str, Dict[str, Any]],
    penalty_model: Dict[str, int],
    effective_units_per_truck: int,
    weather_report: Dict[str, Any],
    corridors: Dict[str, Any],
) -> AllocationResult:
    weather_sla = _compute_weather_sla_violations(weather_report, corridors)

    # ----- problem index -------------------------------------------------
    units = list(enumerate(planning_valid_rows))   # [(u, ReconciledRow), ...]
    days  = list(PLANNING_DAYS)
    EFF   = int(effective_units_per_truck)

    p_tier1 = int(penalty_model["tier1_sla_violation"])
    p_tier2 = int(penalty_model["tier2_sla_violation"])
    p_cold  = int(penalty_model["cold_chain_violation_bonus"])
    p_delay = int(penalty_model["non_sla_day_delay"])

    def unserved_penalty(row: ReconciledRow) -> int:
        sla = p_tier1 if (row.sla_tier or 2) == 1 else p_tier2
        bonus = p_cold if row.is_cold_chain else 0
        return sla + bonus

    def weather_served_penalty(row: ReconciledRow) -> int:
        if not weather_sla.get(row.original.get("corridor_id"), False):
            return 0
        sla = p_tier1 if (row.sla_tier or 2) == 1 else p_tier2
        bonus = p_cold if row.is_cold_chain else 0
        return sla + bonus

    # ----- model ---------------------------------------------------------
    prob = pulp.LpProblem("seewees_dispatch", pulp.LpMinimize)

    # x[u, d] = 1 if unit u placed on day d
    x = {
        (u, d): pulp.LpVariable(f"x_{u}_{d}", cat=pulp.LpBinary)
        for u, _ in units for d in days
    }

    # truck counts per day
    tc = {d: pulp.LpVariable(f"tc_{d}", lowBound=0, cat=pulp.LpInteger) for d in days}
    ts = {d: pulp.LpVariable(f"ts_{d}", lowBound=0, cat=pulp.LpInteger) for d in days}

    # bump indicator: bumped[u] = 1 if served but on the OTHER day
    bumped = {u: pulp.LpVariable(f"bump_{u}", cat=pulp.LpBinary) for u, _ in units}

    # ----- constraints ---------------------------------------------------

    # At most one placement per unit
    for u, _ in units:
        prob += pulp.lpSum(x[u, d] for d in days) <= 1, f"once_{u}"

    # Per-day volume vs truck count
    for d in days:
        prob += (
            pulp.lpSum(x[u, d] for u, row in units if row.is_cold_chain)
            <= EFF * tc[d]
        ), f"cold_volume_{d}"
        prob += (
            pulp.lpSum(x[u, d] for u, row in units if not row.is_cold_chain)
            <= EFF * ts[d]
        ), f"std_volume_{d}"

    # Truck and driver pools per day
    for d in days:
        r = resources_by_day.get(d, {})
        prob += tc[d] <= int(r.get("truck_temp_controlled", 0)), f"reefer_pool_{d}"
        prob += ts[d] <= int(r.get("truck_standard", 0)), f"std_pool_{d}"
        prob += tc[d] + ts[d] <= int(r.get("driver", 0)), f"driver_pool_{d}"

    # bumped[u] >= x[u, other_day] when requested day is known
    for u, row in units:
        requested = row.original.get("planning_day")
        if requested not in days:
            # Unknown requested day -> any day counts as bumped=0 (don't penalize)
            continue
        alt = "Day1" if requested == "Day0" else "Day0"
        prob += bumped[u] >= x[u, alt], f"bump_def_{u}"

    # ----- objective -----------------------------------------------------

    unserved_term = pulp.lpSum(
        unserved_penalty(row) * (1 - pulp.lpSum(x[u, d] for d in days))
        for u, row in units
    )

    delay_term = pulp.lpSum(p_delay * bumped[u] for u, _ in units)

    weather_term = pulp.lpSum(
        weather_served_penalty(row) * pulp.lpSum(x[u, d] for d in days)
        for u, row in units if weather_served_penalty(row) > 0
    )

    # Tie-breaker: when the weather-induced penalty for serving equals the
    # full unserved penalty (which happens when buffer breaches SLA on the
    # ONLY corridor a unit can use), prefer serving over not-serving. The
    # operational truth: a late delivery is strictly better than a missed
    # one. We subtract a tiny epsilon per served unit - small enough that
    # it cannot flip any non-tied decision.
    SERVED_TIE_BREAK = 0.001
    served_bonus = pulp.lpSum(
        SERVED_TIE_BREAK * pulp.lpSum(x[u, d] for d in days)
        for u, _ in units
    )

    prob += unserved_term + delay_term + weather_term - served_bonus

    # ----- solve ---------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    status_code = prob.solve(solver)
    solver_status = pulp.LpStatus[status_code]

    # ----- decode solution ----------------------------------------------
    usage: Dict[str, DayResourceUsage] = {}
    for d in days:
        r = resources_by_day.get(d, {})
        usage[d] = DayResourceUsage(
            day=d,
            drivers_available=int(r.get("driver", 0)),
            truck_standard_available=int(r.get("truck_standard", 0)),
            truck_temp_controlled_available=int(r.get("truck_temp_controlled", 0)),
        )

    assignments: List[UnitAssignment] = []
    for u, row in units:
        requested = row.original.get("planning_day") or ""
        corridor = row.original.get("corridor_id") or ""
        uid = str(row.original.get("unique_item_id"))
        canonical = row.canonical_item_id or "(unresolved)"
        tier = row.sla_tier or 2
        is_cold = bool(row.is_cold_chain)

        chosen_day: Optional[str] = None
        for d in days:
            if pulp.value(x[u, d]) is not None and pulp.value(x[u, d]) > 0.5:
                chosen_day = d
                break

        breakdown: Dict[str, int] = {}
        served = chosen_day is not None
        truck_type: Optional[str] = None
        reason = ""

        if served:
            truck_type = "truck_temp_controlled" if is_cold else "truck_standard"
            usage[chosen_day].units_cold_chain += int(is_cold)
            usage[chosen_day].units_standard += int(not is_cold)
            delay_hit = chosen_day != requested and requested in PLANNING_DAYS
            if delay_hit:
                breakdown["non_sla_day_delay"] = p_delay
                reason = f"Bumped from {requested} to {chosen_day} (ILP-optimal)."
            else:
                reason = f"Dispatched on {chosen_day}."

            if weather_sla.get(corridor, False):
                if tier == 1:
                    breakdown["tier1_sla_violation_weather"] = p_tier1
                else:
                    breakdown["tier2_sla_violation_weather"] = p_tier2
                if is_cold:
                    breakdown["cold_chain_violation_bonus_weather"] = p_cold
                reason += " | Weather buffer breaches SLA cap on this corridor."
        else:
            if tier == 1:
                breakdown["tier1_sla_violation_unserved"] = p_tier1
            else:
                breakdown["tier2_sla_violation_unserved"] = p_tier2
            if is_cold:
                breakdown["cold_chain_violation_bonus_unserved"] = p_cold
            reason = "Unserved: ILP could not fit within resource pools."

        total = sum(breakdown.values())
        assignments.append(UnitAssignment(
            unit_uid=uid,
            corridor_id=corridor,
            canonical_item_id=canonical,
            sla_tier=tier,
            is_cold_chain=is_cold,
            requested_day=requested,
            assigned_day=chosen_day,
            truck_type=truck_type,
            served=served,
            penalty_breakdown=breakdown,
            penalty_total=total,
            reason=reason,
        ))

    # Finalize per-day truck/driver counts from the solver result
    for d in days:
        usage[d].trucks_temp_used = int(round(pulp.value(tc[d]) or 0))
        usage[d].trucks_standard_used = int(round(pulp.value(ts[d]) or 0))
        usage[d].drivers_used = usage[d].trucks_temp_used + usage[d].trucks_standard_used

    # Aggregate. Recompute total from per-assignment breakdowns so the
    # tie-break epsilon does not leak into the displayed integer penalty.
    pb_total: Dict[str, int] = {}
    for a in assignments:
        for k, v in a.penalty_breakdown.items():
            pb_total[k] = pb_total.get(k, 0) + v
    total_penalty = sum(pb_total.values())

    served_count   = sum(1 for a in assignments if a.served)
    unserved_count = len(assignments) - served_count
    tier1_unserved = sum(1 for a in assignments if not a.served and a.sla_tier == 1)
    cold_unserved  = sum(1 for a in assignments if not a.served and a.is_cold_chain)
    bumped_count   = sum(1 for a in assignments if a.served and a.assigned_day != a.requested_day
                         and a.requested_day in PLANNING_DAYS)

    summary = {
        "units_total": len(assignments),
        "units_served": served_count,
        "units_unserved": unserved_count,
        "units_bumped": bumped_count,
        "tier1_unserved": tier1_unserved,
        "cold_chain_unserved": cold_unserved,
        "weather_sla_violated_corridors": [c for c, v in weather_sla.items() if v],
        "solver_status": solver_status,
        "solver_engine": "ILP (PuLP/CBC)",
    }

    return AllocationResult(
        assignments=assignments,
        usage_by_day=usage,
        weather_sla_violations_by_corridor=weather_sla,
        total_penalty=total_penalty,
        penalty_breakdown=pb_total,
        summary=summary,
        solver_status=solver_status,
    )
