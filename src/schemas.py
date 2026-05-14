"""Pydantic schemas for ContextAgent's structured output.

These define the EXACT shape we demand from the LLM when extracting the
SeeWeeS playbook. OpenAI structured outputs will reject any response that
does not match these schemas, eliminating an entire class of LLM failure
modes (missing fields, type drift, extra keys).
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Item Master (Appendix A.1)
# ---------------------------------------------------------------------------

class ItemMasterRow(BaseModel):
    canonical_item_id: str = Field(..., description="Stable analytics key, e.g. 'RMD-100'.")
    item_id: int = Field(..., description="Numeric item_id used in shipment CSV. NOT guaranteed unique across strength variants.")
    canonical_item_name: str = Field(..., description="Full canonical name including strength/variant qualifiers.")
    medicine_type: str
    temp_control: str = Field(..., description="Verbatim temp_control string, e.g. 'Cold (2-8C)' or 'Room Temp (20-25C)'.")
    product_class: str


# ---------------------------------------------------------------------------
# Alias Table (Appendix A.2)
# ---------------------------------------------------------------------------

class AliasRow(BaseModel):
    alias_name: str
    canonical_item_id: str
    confidence_tier: Literal["ALIAS_MATCH"] = "ALIAS_MATCH"
    notes: str = ""


# ---------------------------------------------------------------------------
# Legacy ID Map (Appendix A.3)
# ---------------------------------------------------------------------------

class LegacyMapRow(BaseModel):
    legacy_item_id: int
    canonical_item_id: str
    rule: Literal["LEGACY_ID_MAP", "SPECIAL_CASE"]
    rationale: str


# ---------------------------------------------------------------------------
# Corridor + Waypoint (§3 of playbook)
# ---------------------------------------------------------------------------

class Waypoint(BaseModel):
    waypoint_id: str
    city: str
    lat: float
    lon: float


class CorridorExtracted(BaseModel):
    """Corridor information that exists in the playbook itself.

    Note: sla_tier, base_transit_hours, and max_transit_hours are NOT
    extracted here - those are engineering augmentations layered in via
    augmentations.json. We deliberately don't ask the LLM for them.
    """
    corridor_id: str
    corridor_name: str
    origin_dc: str
    destination_region: str
    default_sla_tier_label: str = Field(..., description="Verbatim from playbook §3.1, e.g. 'Tier 1' or 'Tier 2'.")
    waypoints: List[Waypoint]


# ---------------------------------------------------------------------------
# Weather + Buffers + Penalty + Truck
# ---------------------------------------------------------------------------

class WeatherThresholds(BaseModel):
    """From playbook §6.5.1 (Weather Triggers Daily Index)."""
    heavy_precipitation_mm_day: float
    high_wind_gust_kmh: float
    freezing_temp_c: float


class TravelBufferRow(BaseModel):
    """One row of the playbook §6.5.2 buffer policy table."""
    risk_score: int = Field(..., ge=0, le=3)
    buffer_pct: int = Field(..., ge=0)
    escalation: bool


class TruckModel(BaseModel):
    """From playbook §8 (Truck Capacity & Packing)."""
    capacity_volume_units: int
    packing_inefficiency_buffer_pct: int


class PenaltyModel(BaseModel):
    """From playbook §13 (Resource Constraints and Allocation Policy)."""
    tier1_sla_violation: int
    tier2_sla_violation: int
    cold_chain_violation_bonus: int
    non_sla_day_delay: int


class SlaTierRow(BaseModel):
    """From playbook §7 (Dispatch SLA Classes)."""
    sla_tier_label: str = Field(..., description="Verbatim, e.g. 'Tier 1'.")
    sla_tier_number: int
    medicine_category: str
    max_time_in_transit_hours: float


# ---------------------------------------------------------------------------
# Appendix A.5 — unique_item_id regex patterns per product_class
# ---------------------------------------------------------------------------

class UidRegexRow(BaseModel):
    """One row of Appendix A.5 (Identifier Format Rules).

    Used to (a) validate existing unique_item_id values, and
    (b) optionally regenerate missing UIDs as placeholders.
    """
    product_class: str = Field(..., description="Matches product_class in item_master, e.g. 'Antiviral'.")
    expected_regex: str = Field(..., description="Verbatim regex pattern, e.g. '^RMD-\\d{4}-\\d{4}$'.")
    example: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Top-level extraction envelope
# ---------------------------------------------------------------------------

class PlaybookExtraction(BaseModel):
    """Full structured extraction of the SeeWeeS playbook by ContextAgent."""
    item_master: List[ItemMasterRow]
    aliases: List[AliasRow]
    legacy_map: List[LegacyMapRow]
    corridors: List[CorridorExtracted]
    weather_thresholds: WeatherThresholds
    travel_buffer_by_risk: List[TravelBufferRow]
    truck: TruckModel
    penalty_model: PenaltyModel
    sla_tiers: List[SlaTierRow]
    uid_regex_rules: List[UidRegexRow]
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Optional: short note on any ambiguities encountered.",
    )


# ---------------------------------------------------------------------------
# AuditReport (output of the AuditorAgent / deterministic checks)
# ---------------------------------------------------------------------------

class AuditFinding(BaseModel):
    """A single audit observation about the dispatch plan."""
    check_id: str = Field(..., description="Stable ID like 'A-ESC-01' (escalation), 'A-CC-01' (cold-chain integrity), etc.")
    title: str = Field(..., description="Short human-readable title.")
    status: Literal["pass", "warn", "fail"]
    severity: Literal["info", "low", "medium", "high", "critical"]
    description: str = Field(..., description="What the auditor observed, citing specific numbers.")
    recommendation: str = Field(..., description="Concrete next action for the operations manager.")


class AuditReport(BaseModel):
    """Full compliance audit of the allocator's output."""
    overall_status: Literal["pass", "warn", "fail"]
    headline: str = Field(..., description="One-sentence executive summary of audit outcome.")
    findings: List[AuditFinding]
    rules_evaluated: int
    rules_passed: int
    rules_failed: int
    rules_warned: int
