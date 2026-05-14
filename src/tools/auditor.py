"""Compliance auditor for the dispatch plan.

Runs a battery of DETERMINISTIC checks against the allocator output and the
playbook-derived state, then hands the raw findings to AuditorAgent (an LLM)
to compose executive-friendly narrative descriptions and recommendations.

Why a deterministic core + LLM narrative?
    - The pass/fail status of every rule must be reproducible (no LLM drift).
    - But the language ("what does this mean for the VP?") is exactly what
      LLMs are good at.

Checks implemented (each emits a single AuditFinding):
    A-ESC-01  Weather risk-score 3 must trigger escalation flag
    A-WX-01   Weather-induced SLA breach detection (per-corridor)
    A-CC-01   Cold-chain integrity: every cold unit assigned to reefer
    A-CAP-01  Capacity math: trucks/drivers used <= available (per-day)
    A-TIER-01 SLA tier consistency: every assignment.sla_tier matches its corridor
    A-UNSRV-01 Unserved Tier-1 sanity (should be zero unless capacity truly exhausted)
    A-DQ-01   Data quality regeneration usage (info-level)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from schemas import AuditFinding


@dataclass
class CheckContext:
    """Everything an audit check needs to make a decision."""
    weather: Dict[str, Any]
    sla_by_corridor: Dict[str, Any]
    allocation: Any   # AllocationResult, kept loose to avoid circular import
    corridors: Dict[str, Any]
    constants: Dict[str, Any]
    planning_summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Individual checks. Each returns a dict (pre-AuditFinding) so the LLM can
# enrich `description` and `recommendation` afterwards. We pre-populate
# sensible defaults so the result is usable even without the LLM step.
# ---------------------------------------------------------------------------

def _check_escalation(ctx: CheckContext) -> Dict[str, Any]:
    bad = [cid for cid, w in ctx.weather.items() if w.get("max_48h_risk_score") == 3]
    if not bad:
        return {
            "check_id": "A-ESC-01",
            "title": "Weather escalation threshold",
            "status": "pass",
            "severity": "info",
            "description": "No corridor reached risk-score 3 in the 48h window; no escalation required.",
            "recommendation": "Continue monitoring forecasts; no action needed.",
        }
    cids = ", ".join(bad)
    return {
        "check_id": "A-ESC-01",
        "title": "Weather escalation threshold",
        "status": "fail",
        "severity": "critical",
        "description": f"Corridor(s) {cids} reached weather risk-score 3 (playbook §6.5.2). Escalation required.",
        "recommendation": "Escalate to operations leadership immediately; activate contingency dispatch protocol.",
    }


def _check_weather_sla(ctx: CheckContext) -> Dict[str, Any]:
    breaches = {
        cid: sla for cid, sla in ctx.sla_by_corridor.items()
        if sla.get("sla_violation")
    }
    if not breaches:
        return {
            "check_id": "A-WX-01",
            "title": "Weather-induced SLA breach",
            "status": "pass",
            "severity": "info",
            "description": "Weather buffers keep all corridors within their SLA caps.",
            "recommendation": "No action; continue as planned.",
        }
    parts = []
    for cid, sla in breaches.items():
        tier = ctx.corridors.get(cid, {}).get("sla_tier", "?")
        parts.append(
            f"{cid} (Tier {tier}): adjusted {sla.get('adjusted_transit_hours')}h > cap {sla.get('max_transit_hours')}h "
            f"(headroom {sla.get('headroom_hours')}h)"
        )
    return {
        "check_id": "A-WX-01",
        "title": "Weather-induced SLA breach",
        "status": "fail",
        "severity": "high",
        "description": "Weather buffer pushes transit time above SLA cap on: " + "; ".join(parts),
        "recommendation": "Consider earlier Day-0 dispatch, alternate corridor, or pre-positioning to reduce exposed transit hours.",
    }


def _check_cold_chain_integrity(ctx: CheckContext) -> Dict[str, Any]:
    bad = []
    for a in ctx.allocation.assignments:
        if a.served and a.is_cold_chain and a.truck_type != "truck_temp_controlled":
            bad.append(a.unit_uid)
    if not bad:
        return {
            "check_id": "A-CC-01",
            "title": "Cold-chain integrity",
            "status": "pass",
            "severity": "info",
            "description": "Every served cold-chain unit is assigned to a temp-controlled truck.",
            "recommendation": "No action; cold-chain integrity is intact.",
        }
    return {
        "check_id": "A-CC-01",
        "title": "Cold-chain integrity",
        "status": "fail",
        "severity": "critical",
        "description": f"{len(bad)} cold-chain unit(s) assigned to standard trucks: {bad[:5]}{'...' if len(bad) > 5 else ''}.",
        "recommendation": "Re-route these units to a temp-controlled truck before dispatch; review allocator logic.",
    }


def _check_capacity_math(ctx: CheckContext) -> Dict[str, Any]:
    violations = []
    for day, u in ctx.allocation.usage_by_day.items():
        if u.trucks_standard_used > u.truck_standard_available:
            violations.append(f"{day} standard trucks {u.trucks_standard_used}/{u.truck_standard_available}")
        if u.trucks_temp_used > u.truck_temp_controlled_available:
            violations.append(f"{day} temp-controlled trucks {u.trucks_temp_used}/{u.truck_temp_controlled_available}")
        if u.drivers_used > u.drivers_available:
            violations.append(f"{day} drivers {u.drivers_used}/{u.drivers_available}")
    if not violations:
        return {
            "check_id": "A-CAP-01",
            "title": "Resource capacity math",
            "status": "pass",
            "severity": "info",
            "description": "All resource usage (drivers, standard trucks, temp-controlled trucks) is within daily availability.",
            "recommendation": "No action.",
        }
    return {
        "check_id": "A-CAP-01",
        "title": "Resource capacity math",
        "status": "fail",
        "severity": "critical",
        "description": "Capacity overshoot: " + "; ".join(violations),
        "recommendation": "Re-run allocator with corrected pools; investigate solver math.",
    }


def _check_tier_consistency(ctx: CheckContext) -> Dict[str, Any]:
    mismatched = []
    for a in ctx.allocation.assignments:
        expected = ctx.corridors.get(a.corridor_id, {}).get("sla_tier")
        if expected is not None and a.sla_tier != expected:
            mismatched.append((a.unit_uid, a.corridor_id, a.sla_tier, expected))
    if not mismatched:
        return {
            "check_id": "A-TIER-01",
            "title": "SLA tier consistency",
            "status": "pass",
            "severity": "info",
            "description": "Every assignment's SLA tier matches its corridor configuration.",
            "recommendation": "No action.",
        }
    return {
        "check_id": "A-TIER-01",
        "title": "SLA tier consistency",
        "status": "fail",
        "severity": "medium",
        "description": f"{len(mismatched)} unit(s) have an SLA tier inconsistent with their corridor.",
        "recommendation": "Investigate corridor/tier mapping in augmentations.json.",
    }


def _check_unserved_tier1(ctx: CheckContext) -> Dict[str, Any]:
    summary = ctx.allocation.summary
    unserved = summary.get("tier1_unserved", 0)
    cold_unserved = summary.get("cold_chain_unserved", 0)
    if unserved == 0 and cold_unserved == 0:
        return {
            "check_id": "A-UNSRV-01",
            "title": "Unserved high-priority units",
            "status": "pass",
            "severity": "info",
            "description": "No Tier-1 or cold-chain units left unserved.",
            "recommendation": "No action.",
        }
    return {
        "check_id": "A-UNSRV-01",
        "title": "Unserved high-priority units",
        "status": "warn" if unserved <= 2 else "fail",
        "severity": "high",
        "description": f"{unserved} Tier-1 unit(s) unserved; {cold_unserved} cold-chain unit(s) unserved.",
        "recommendation": "Surface to leadership; consider expedited courier or interday rebalancing.",
    }


def _check_dq_regeneration(ctx: CheckContext) -> Dict[str, Any]:
    by_reason = ctx.planning_summary.get("by_reason_code", {})
    n_generated = by_reason.get("generated_identifier", 0)
    if n_generated == 0:
        return {
            "check_id": "A-DQ-01",
            "title": "Data-quality regeneration usage",
            "status": "pass",
            "severity": "info",
            "description": "No planning-window rows required Appendix A.5 unique_item_id regeneration.",
            "recommendation": "No action.",
        }
    return {
        "check_id": "A-DQ-01",
        "title": "Data-quality regeneration usage",
        "status": "warn",
        "severity": "low",
        "description": (
            f"{n_generated} planning-window row(s) had missing unique_item_id and were rescued via A.5 "
            "placeholder generation. Generated IDs follow the GEN### convention."
        ),
        "recommendation": "Reconcile generated IDs with physical inventory after dispatch; investigate upstream data source.",
    }


CHECKS = [
    _check_escalation,
    _check_weather_sla,
    _check_cold_chain_integrity,
    _check_capacity_math,
    _check_tier_consistency,
    _check_unserved_tier1,
    _check_dq_regeneration,
]


def run_deterministic_audit(ctx: CheckContext) -> List[Dict[str, Any]]:
    """Run every check and return a list of finding dicts (one per check)."""
    return [chk(ctx) for chk in CHECKS]
