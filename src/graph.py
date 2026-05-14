"""LangGraph for the SeeWeeS multi-corridor dispatch pipeline.

Topology:

                       load_playbook
                            |
            +---------------+---------------+
            |               |               |
        ingest         weather        resources           (parallel fan-out)
            |               |               |
            +-------+-------+-------+-------+
                            |
                          kpis
                            |
                        allocate
                            |
                         report
                            |
                          email
                            |
                           END

`ingest`, `weather`, and `resources` are independent and execute in parallel
because LangGraph fans out when multiple edges leave the same source.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, END

from tools.playbook_loader import load_playbook, Playbook
from tools.context_extractor import extract_and_persist
from tools.csv_tools import load_and_reconcile_shipments, ShipmentDataset
from tools.weather_tools import evaluate_corridor_weather, compute_sla_violation_from_weather
from tools.resource_loader import load_resource_availability
from tools.kpi_engine import compute_corridor_kpis, compute_overall_kpis
from tools.allocator import allocate, AllocationResult
from tools.auditor import CheckContext, run_deterministic_audit
from tools.email_tools import send_email_smtp
from agents import run_auditor_agent, run_report_agent
from schemas import AuditReport, PlaybookExtraction


class AppState(TypedDict, total=False):
    # Inputs
    playbook_md_path: str
    augmented_dir: str
    shipments_csv_path: str
    resources_csv_path: str
    force_re_extract: bool

    # Stage outputs
    context_extraction: PlaybookExtraction
    playbook: Playbook
    shipments: ShipmentDataset
    weather: Dict[str, Any]
    sla_by_corridor: Dict[str, Any]
    resources: Dict[str, Any]
    corridor_kpis: Dict[str, Any]
    overall_kpis: Dict[str, Any]
    allocation: AllocationResult
    audit_report: AuditReport

    # Final
    report_html: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_context_extraction(state: AppState) -> AppState:
    """ContextAgent: read the SeeWeeS Specialty Dispatch Playbook (markdown) and
    emit a structured PlaybookExtraction (validated via pydantic + OpenAI
    structured outputs). The result is cached to data/augmented/extracted_context.json."""
    extraction = extract_and_persist(
        playbook_md_path=state["playbook_md_path"],
        augmented_dir=state["augmented_dir"],
        force=state.get("force_re_extract", False),
    )
    return {"context_extraction": extraction}


def node_load_playbook(state: AppState) -> AppState:
    """Merge the LLM-extracted playbook content with the hand-curated
    engineering augmentations (sla_tier per corridor, base_transit_hours,
    effective_units_per_truck) into the unified Playbook object the rest of
    the pipeline consumes."""
    pb = load_playbook(
        state["playbook_md_path"],
        state["augmented_dir"],
        force_re_extract=False,  # already done in node_context_extraction
    )
    return {"playbook": pb}


def node_ingest(state: AppState) -> AppState:
    ds = load_and_reconcile_shipments(state["shipments_csv_path"], state["playbook"])
    return {"shipments": ds}


def node_weather(state: AppState) -> AppState:
    pb: Playbook = state["playbook"]
    w = evaluate_corridor_weather(
        pb.corridors,
        pb.constants["weather_thresholds"],
        pb.constants["travel_buffer_by_risk"],
    )
    sla_by_corridor = {
        cid: compute_sla_violation_from_weather(pb.corridors[cid], w[cid]["buffer_pct"])
        for cid in pb.corridors
    }
    return {"weather": w, "sla_by_corridor": sla_by_corridor}


def node_resources(state: AppState) -> AppState:
    return {"resources": load_resource_availability(state["resources_csv_path"])}


def node_kpis(state: AppState) -> AppState:
    pb: Playbook = state["playbook"]
    ds: ShipmentDataset = state["shipments"]
    corridor_kpis = compute_corridor_kpis(ds.planning, ds.history, pb.corridor_ids())
    overall_kpis = compute_overall_kpis(ds.planning, ds.history)
    return {"corridor_kpis": corridor_kpis, "overall_kpis": overall_kpis}


def node_join(state: AppState) -> AppState:
    """No-op barrier that joins the three parallel branches (kpis, weather,
    resources) into a single super-step before allocation. Prevents downstream
    nodes from being scheduled multiple times due to differing branch depths."""
    return {}


def node_allocate(state: AppState) -> AppState:
    pb: Playbook = state["playbook"]
    ds: ShipmentDataset = state["shipments"]
    alloc = allocate(
        planning_valid_rows=ds.planning.valid_rows,
        resources_by_day=state["resources"],
        penalty_model=pb.constants["penalty_model"],
        effective_units_per_truck=pb.constants["truck"]["effective_units_per_truck"],
        weather_report=state["weather"],
        corridors=pb.corridors,
    )
    return {"allocation": alloc}


def node_audit(state: AppState) -> AppState:
    """Compliance audit: deterministic checks -> LLM narrative.

    The deterministic engine produces pass/warn/fail findings with locked
    statuses. AuditorAgent rewrites the descriptions and recommendations for
    an executive audience and stamps the overall status."""
    pb: Playbook = state["playbook"]
    ds: ShipmentDataset = state["shipments"]
    alloc: AllocationResult = state["allocation"]
    ctx = CheckContext(
        weather=state["weather"],
        sla_by_corridor=state["sla_by_corridor"],
        allocation=alloc,
        corridors=pb.corridors,
        constants=pb.constants,
        planning_summary=ds.planning.summary(),
    )
    findings = run_deterministic_audit(ctx)
    plan_context = {
        "allocation_summary": alloc.summary,
        "total_penalty": alloc.total_penalty,
        "penalty_breakdown": alloc.penalty_breakdown,
        "weather_max_risk_by_corridor": {
            cid: w.get("max_48h_risk_score") for cid, w in state["weather"].items()
        },
        "sla_by_corridor": state["sla_by_corridor"],
    }
    audit_report = run_auditor_agent(findings, plan_context)
    return {"audit_report": audit_report}


def node_report(state: AppState) -> AppState:
    pb: Playbook = state["playbook"]
    ds: ShipmentDataset = state["shipments"]
    alloc: AllocationResult = state["allocation"]

    audit_report: AuditReport = state.get("audit_report")  # type: ignore[assignment]

    # Build a flat, JSON-serializable payload for the LLM. Every number that
    # may appear in the report must be in here - the LLM is instructed to
    # cite verbatim and not invent.
    payload = {
        "audit_report": audit_report.model_dump() if audit_report else None,
        "corridors": pb.corridors,
        "constants": pb.constants,
        "weather": state["weather"],
        "sla_by_corridor": state["sla_by_corridor"],
        "resources": state["resources"],
        "corridor_kpis": state["corridor_kpis"],
        "overall_kpis": state["overall_kpis"],
        "allocation_summary": alloc.summary,
        "allocation_total_penalty": alloc.total_penalty,
        "allocation_penalty_breakdown": alloc.penalty_breakdown,
        "allocation_usage_by_day": {d: {
            "drivers_used": u.drivers_used,
            "drivers_available": u.drivers_available,
            "trucks_standard_used": u.trucks_standard_used,
            "trucks_standard_available": u.truck_standard_available,
            "trucks_temp_used": u.trucks_temp_used,
            "trucks_temp_available": u.truck_temp_controlled_available,
            "units_standard": u.units_standard,
            "units_cold_chain": u.units_cold_chain,
        } for d, u in alloc.usage_by_day.items()},
        "weather_sla_violations_by_corridor": alloc.weather_sla_violations_by_corridor,
        "dq_summary": {
            "history": ds.history.summary(),
            "planning": ds.planning.summary(),
            "planning_excluded_examples": [
                {
                    "corridor_id": r.original.get("corridor_id"),
                    "item_id":     r.original.get("item_id"),
                    "item_name":   r.original.get("item_name"),
                    "shipment_date": str(r.original.get("shipment_date")),
                    "reason_code": r.reason_code,
                    "dq_rule":     r.dq_rule,
                    "notes":       r.notes,
                }
                for r in ds.planning.excluded_rows
            ],
            "planning_legacy_id_matches": [
                {
                    "corridor_id": r.original.get("corridor_id"),
                    "item_id":     r.original.get("item_id"),
                    "canonical_item_id": r.canonical_item_id,
                    "notes":       r.notes,
                }
                for r in ds.planning.rows if r.reason_code == "legacy_id_map"
            ],
            "planning_alias_matches": [
                {
                    "corridor_id": r.original.get("corridor_id"),
                    "name_raw":    r.original.get("item_name"),
                    "canonical_item_id": r.canonical_item_id,
                }
                for r in ds.planning.rows if r.reason_code == "alias_match"
            ],
            "planning_generated_ids": [
                {
                    "corridor_id": r.original.get("corridor_id"),
                    "planning_day": r.original.get("planning_day"),
                    "generated_uid": r.original.get("unique_item_id"),
                    "canonical_item_id": r.canonical_item_id,
                    "product_class": r.product_class,
                    "notes": r.notes,
                }
                for r in ds.planning.rows if r.reason_code == "generated_identifier"
            ],
        },
        "unserved_units": [
            a.to_dict() for a in alloc.assignments if not a.served
        ],
        "bumped_units": [
            a.to_dict() for a in alloc.assignments if a.served and a.assigned_day != a.requested_day
        ],
    }

    html = run_report_agent(payload)
    return {"report_html": html}


def node_email(state: AppState) -> AppState:
    to_email = os.getenv("REPORT_EMAIL_TO", "").strip()
    if not to_email:
        print("REPORT_EMAIL_TO not set -> skipping email send.")
        return {}
    send_email_smtp(
        subject="SeeWeeS Multi-Corridor Dispatch Report (48h)",
        html_body=state.get("report_html", ""),
        to_email=to_email,
    )
    return {}


# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AppState)

    g.add_node("context_extraction", node_context_extraction)
    g.add_node("load_playbook", node_load_playbook)
    g.add_node("ingest", node_ingest)
    g.add_node("weather", node_weather)
    g.add_node("resources", node_resources)
    g.add_node("join", node_join)
    g.add_node("kpis", node_kpis)
    g.add_node("allocate", node_allocate)
    g.add_node("audit", node_audit)
    g.add_node("report", node_report)
    g.add_node("email", node_email)

    g.set_entry_point("context_extraction")
    g.add_edge("context_extraction", "load_playbook")

    # Parallel fan-out (all three branches share the same depth = 1)
    g.add_edge("load_playbook", "ingest")
    g.add_edge("load_playbook", "weather")
    g.add_edge("load_playbook", "resources")

    # Barrier node joins all three branches in a single super-step.
    # KPIs and allocation run sequentially after the join.
    g.add_edge("ingest", "join")
    g.add_edge("weather", "join")
    g.add_edge("resources", "join")
    g.add_edge("join", "kpis")
    g.add_edge("kpis", "allocate")

    g.add_edge("allocate", "audit")
    g.add_edge("audit", "report")
    g.add_edge("report", "email")
    g.add_edge("email", END)

    return g.compile()
