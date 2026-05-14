"""Tests for the deterministic core of the SeeWeeS dispatch pipeline.

These tests cover everything that does NOT require an LLM call or network
access (no OpenAI, no Open-Meteo). They validate the data-quality reconciler,
the allocator's penalty math, and every auditor check. The cached ContextAgent
extraction (data/augmented/extracted_context.json) is used in lieu of a live
LLM call so the suite runs in seconds and is reproducible.

Run from project root:
    pytest -v tests/
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tools.playbook_loader import load_playbook                     # noqa: E402
from tools.csv_tools import load_and_reconcile_shipments            # noqa: E402
from tools.dq_reconciler import (                                   # noqa: E402
    reconcile_shipments,
    REASON_EXACT_MATCH,
    REASON_ALIAS_MATCH,
    REASON_LEGACY_ID_MAP,
    REASON_GENERATED_IDENTIFIER,
    REASON_EXCLUDED_MISSING_UID,
)
from tools.resource_loader import load_resource_availability        # noqa: E402
from tools.allocator import allocate                                # noqa: E402
from tools.auditor import CheckContext, run_deterministic_audit     # noqa: E402


PLAYBOOK_MD = ROOT / "data-for-enhancement" / "SeeWeeS Specialty Dispatch Playbook.md"
AUGMENTED_DIR = ROOT / "data" / "augmented"
SHIPMENTS_CSV = ROOT / "data-for-enhancement" / "Incoming_shipments_14d_multi_corridor.csv"
RESOURCES_CSV = ROOT / "data-for-enhancement" / "Resource_availability_48h.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def playbook():
    """Cached ContextAgent extraction merged with augmentations."""
    assert (AUGMENTED_DIR / "extracted_context.json").exists(), (
        "extracted_context.json missing - run `python src/main.py` once to populate it."
    )
    return load_playbook(PLAYBOOK_MD, AUGMENTED_DIR, force_re_extract=False)


@pytest.fixture(scope="session")
def shipments(playbook):
    return load_and_reconcile_shipments(str(SHIPMENTS_CSV), playbook)


@pytest.fixture(scope="session")
def resources():
    return load_resource_availability(str(RESOURCES_CSV))


@pytest.fixture(scope="session")
def stub_weather_max_risk_2():
    """Fake weather report - max_48h_risk_score=2 on C1, 1 on C2 (matches
    the real-world signal in the project but doesn't require a live API call)."""
    return {
        "C1_I95_NJ_BOS": {
            "corridor_name": "NJ -> Boston (I-95)",
            "Day0": {"risk_score": 1, "worst_waypoint": "C1_W1", "flags": {}},
            "Day1": {"risk_score": 2, "worst_waypoint": "C1_W4", "flags": {}},
            "max_48h_risk_score": 2,
            "buffer_pct": 25,
            "escalation_required": False,
        },
        "C2_NJ_PHL": {
            "corridor_name": "NJ -> Philadelphia",
            "Day0": {"risk_score": 1, "worst_waypoint": "C2_W1", "flags": {}},
            "Day1": {"risk_score": 1, "worst_waypoint": "C2_W1", "flags": {}},
            "max_48h_risk_score": 1,
            "buffer_pct": 10,
            "escalation_required": False,
        },
    }


# ---------------------------------------------------------------------------
# Playbook extraction sanity
# ---------------------------------------------------------------------------

class TestPlaybookExtraction:
    def test_item_master_has_all_canonical_ids(self, playbook):
        canonical_ids = {r["canonical_item_id"] for r in playbook.item_master}
        # 11 canonical entries per Appendix A.1
        assert len(canonical_ids) == 11
        for expected in ["RMD-100", "RMD-200", "INS-LIS", "PMB-KEY", "EPI-AI",
                         "HEP-SOD", "MOR-SUL", "ALB-INH", "EXP-ONC-CT",
                         "LEV-INH", "INS-ASP"]:
            assert expected in canonical_ids

    def test_aliases_extracted(self, playbook):
        assert len(playbook.aliases) == 7

    def test_legacy_map_extracted(self, playbook):
        assert len(playbook.legacy_map) == 4

    def test_corridors_have_engineering_augmentations(self, playbook):
        for cid in ["C1_I95_NJ_BOS", "C2_NJ_PHL"]:
            c = playbook.corridors[cid]
            assert "sla_tier" in c
            assert "max_transit_hours" in c
            assert "base_transit_hours" in c
            assert len(c["waypoints"]) >= 4

    def test_penalty_model_values(self, playbook):
        p = playbook.constants["penalty_model"]
        assert p["tier1_sla_violation"] == 100
        assert p["tier2_sla_violation"] == 40
        assert p["cold_chain_violation_bonus"] == 80
        assert p["non_sla_day_delay"] == 10

    def test_uid_regex_rules_extracted(self, playbook):
        classes = {r["product_class"] for r in playbook.uid_regex_rules}
        for expected in ["Antiviral", "Oncology Biologic", "Emergency",
                         "Controlled", "Respiratory", "Clinical Trial"]:
            assert expected in classes


# ---------------------------------------------------------------------------
# DQ reconciler
# ---------------------------------------------------------------------------

class TestDqReconciler:
    def test_planning_window_fully_dispatchable_with_a5(self, shipments):
        """With A.5 regeneration on, every planning row should be dispatchable."""
        s = shipments.planning.summary()
        assert s["total_rows"] == 33
        assert s["valid_for_dispatch"] == 33
        assert s["excluded"] == 0

    def test_history_summary(self, shipments):
        s = shipments.history.summary()
        assert s["total_rows"] == 96
        assert s["valid_for_dispatch"] == 96  # 94 + 2 rescued

    def test_legacy_ids_resolved(self, shipments):
        legacy_rows = [r for r in shipments.planning.rows
                       if r.reason_code == REASON_LEGACY_ID_MAP]
        item_ids = {r.original["item_id"] for r in legacy_rows}
        # 1070 (-> ALB-INH) and 99999 (-> EXP-ONC-CT) appear in planning window
        assert 1070 in item_ids
        assert 99999 in item_ids

    def test_alias_matches_normalize_whitespace(self, shipments):
        """'Remdesivir 100 mg' (extra space) should map to RMD-100."""
        rmd100_alias = [r for r in shipments.planning.rows
                        if r.reason_code == REASON_ALIAS_MATCH
                        and r.canonical_item_id == "RMD-100"]
        assert len(rmd100_alias) >= 1

    def test_generated_uids_follow_format(self, shipments):
        gen = [r for r in shipments.planning.rows
               if r.reason_code == REASON_GENERATED_IDENTIFIER]
        assert len(gen) == 3  # 3 missing UIDs in the planning window
        for r in gen:
            uid = r.original["unique_item_id"]
            assert "-GEN" in uid
            assert r.dq_rule == "DQ-01-RESCUED-A5"

    def test_strength_variant_disambiguation(self, shipments):
        """item_id=10021 must split into RMD-100 vs RMD-200 by item_name."""
        rmd100 = [r for r in shipments.planning.rows
                  if r.canonical_item_id == "RMD-100"]
        rmd200 = [r for r in shipments.planning.rows
                  if r.canonical_item_id == "RMD-200"]
        assert len(rmd100) > 0
        assert len(rmd200) > 0

    def test_disable_a5_falls_back_to_exclusion(self, playbook, shipments):
        """Sanity: when A.5 regen is off, missing-UID rows are excluded."""
        rows = [r.original for r in shipments.planning.rows]
        # Re-add the original NaN unique_item_ids that we replaced with GEN IDs
        for r in shipments.planning.rows:
            if r.reason_code == REASON_GENERATED_IDENTIFIER:
                rows[shipments.planning.rows.index(r)]["unique_item_id"] = None

        res = reconcile_shipments(
            rows,
            playbook.item_master, playbook.aliases, playbook.legacy_map,
            playbook.corridors,
            uid_regex_rules=playbook.uid_regex_rules,
            dq_config={"enable_uid_regeneration_a5": False},
        )
        n_excluded = sum(1 for r in res.rows
                         if r.reason_code == REASON_EXCLUDED_MISSING_UID)
        assert n_excluded == 3


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

class TestAllocator:
    def test_serves_all_planning_units(self, playbook, shipments, resources, stub_weather_max_risk_2):
        alloc = allocate(
            planning_valid_rows=shipments.planning.valid_rows,
            resources_by_day=resources,
            penalty_model=playbook.constants["penalty_model"],
            effective_units_per_truck=playbook.constants["truck"]["effective_units_per_truck"],
            weather_report=stub_weather_max_risk_2,
            corridors=playbook.corridors,
        )
        assert alloc.summary["units_total"] == 33
        assert alloc.summary["units_served"] == 33
        assert alloc.summary["units_unserved"] == 0

    def test_capacity_within_limits(self, playbook, shipments, resources, stub_weather_max_risk_2):
        alloc = allocate(
            planning_valid_rows=shipments.planning.valid_rows,
            resources_by_day=resources,
            penalty_model=playbook.constants["penalty_model"],
            effective_units_per_truck=playbook.constants["truck"]["effective_units_per_truck"],
            weather_report=stub_weather_max_risk_2,
            corridors=playbook.corridors,
        )
        for day, u in alloc.usage_by_day.items():
            assert u.trucks_standard_used <= u.truck_standard_available
            assert u.trucks_temp_used <= u.truck_temp_controlled_available
            assert u.drivers_used <= u.drivers_available

    def test_cold_chain_routed_to_reefer(self, playbook, shipments, resources, stub_weather_max_risk_2):
        alloc = allocate(
            planning_valid_rows=shipments.planning.valid_rows,
            resources_by_day=resources,
            penalty_model=playbook.constants["penalty_model"],
            effective_units_per_truck=playbook.constants["truck"]["effective_units_per_truck"],
            weather_report=stub_weather_max_risk_2,
            corridors=playbook.corridors,
        )
        for a in alloc.assignments:
            if a.served and a.is_cold_chain:
                assert a.truck_type == "truck_temp_controlled"

    def test_weather_sla_penalty_applied_on_c1(self, playbook, shipments, resources, stub_weather_max_risk_2):
        alloc = allocate(
            planning_valid_rows=shipments.planning.valid_rows,
            resources_by_day=resources,
            penalty_model=playbook.constants["penalty_model"],
            effective_units_per_truck=playbook.constants["truck"]["effective_units_per_truck"],
            weather_report=stub_weather_max_risk_2,
            corridors=playbook.corridors,
        )
        # C1 base=5h, buffer=25% -> adjusted=6.25h, cap=6h -> violation
        assert "C1_I95_NJ_BOS" in alloc.summary["weather_sla_violated_corridors"]
        # Penalty breakdown should include weather-driven Tier-1 violations
        assert alloc.penalty_breakdown.get("tier1_sla_violation_weather", 0) > 0
        # No unserved penalties this run
        assert alloc.penalty_breakdown.get("tier1_sla_violation_unserved", 0) == 0


# ---------------------------------------------------------------------------
# Auditor - deterministic checks
# ---------------------------------------------------------------------------

class TestAuditor:
    @pytest.fixture
    def audit_ctx(self, playbook, shipments, resources, stub_weather_max_risk_2):
        alloc = allocate(
            planning_valid_rows=shipments.planning.valid_rows,
            resources_by_day=resources,
            penalty_model=playbook.constants["penalty_model"],
            effective_units_per_truck=playbook.constants["truck"]["effective_units_per_truck"],
            weather_report=stub_weather_max_risk_2,
            corridors=playbook.corridors,
        )
        sla_by_corridor = {
            cid: {
                "base_transit_hours": playbook.corridors[cid]["base_transit_hours"],
                "max_transit_hours": playbook.corridors[cid]["max_transit_hours"],
                "adjusted_transit_hours": playbook.corridors[cid]["base_transit_hours"] * (1 + w["buffer_pct"] / 100),
                "headroom_hours": playbook.corridors[cid]["max_transit_hours"]
                                  - playbook.corridors[cid]["base_transit_hours"] * (1 + w["buffer_pct"] / 100),
                "sla_violation": (playbook.corridors[cid]["base_transit_hours"] * (1 + w["buffer_pct"] / 100))
                                 > playbook.corridors[cid]["max_transit_hours"],
            }
            for cid, w in stub_weather_max_risk_2.items()
        }
        return CheckContext(
            weather=stub_weather_max_risk_2,
            sla_by_corridor=sla_by_corridor,
            allocation=alloc,
            corridors=playbook.corridors,
            constants=playbook.constants,
            planning_summary=shipments.planning.summary(),
        )

    def test_all_seven_checks_run(self, audit_ctx):
        findings = run_deterministic_audit(audit_ctx)
        assert len(findings) == 7
        ids = {f["check_id"] for f in findings}
        for expected in ["A-ESC-01", "A-WX-01", "A-CC-01", "A-CAP-01",
                         "A-TIER-01", "A-UNSRV-01", "A-DQ-01"]:
            assert expected in ids

    def test_escalation_pass_at_risk_2(self, audit_ctx):
        findings = run_deterministic_audit(audit_ctx)
        esc = next(f for f in findings if f["check_id"] == "A-ESC-01")
        # Max risk is 2 in our stub, escalation triggers only at 3
        assert esc["status"] == "pass"

    def test_weather_sla_fail_on_c1(self, audit_ctx):
        findings = run_deterministic_audit(audit_ctx)
        wx = next(f for f in findings if f["check_id"] == "A-WX-01")
        assert wx["status"] == "fail"
        assert wx["severity"] == "high"
        assert "C1_I95_NJ_BOS" in wx["description"]

    def test_cold_chain_integrity_pass(self, audit_ctx):
        findings = run_deterministic_audit(audit_ctx)
        cc = next(f for f in findings if f["check_id"] == "A-CC-01")
        assert cc["status"] == "pass"

    def test_capacity_math_pass(self, audit_ctx):
        findings = run_deterministic_audit(audit_ctx)
        cap = next(f for f in findings if f["check_id"] == "A-CAP-01")
        assert cap["status"] == "pass"

    def test_dq_regeneration_warns_when_uids_generated(self, audit_ctx):
        findings = run_deterministic_audit(audit_ctx)
        dq = next(f for f in findings if f["check_id"] == "A-DQ-01")
        assert dq["status"] == "warn"
        assert "3" in dq["description"]

    def test_escalation_fires_when_risk_score_3(self, audit_ctx):
        """Mutate the stub to risk=3 and confirm escalation flips to fail."""
        audit_ctx.weather["C1_I95_NJ_BOS"]["max_48h_risk_score"] = 3
        findings = run_deterministic_audit(audit_ctx)
        esc = next(f for f in findings if f["check_id"] == "A-ESC-01")
        assert esc["status"] == "fail"
        assert esc["severity"] == "critical"
        # Reset
        audit_ctx.weather["C1_I95_NJ_BOS"]["max_48h_risk_score"] = 2
