"""Load the playbook into a unified Playbook object.

Source layout:
    - data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md
        -> the canonical document. Read by ContextAgent (see context_extractor.py).
    - data/augmented/extracted_context.json
        -> produced by ContextAgent. Contains every value transcribed from
           the playbook (item master, aliases, legacy, weather, penalty, etc.).
    - data/augmented/augmentations.json
        -> hand-curated engineering decisions NOT present in the playbook
           (sla_tier per corridor, base_transit_hours, effective_units_per_truck).

This loader merges the two into the legacy Playbook dataclass shape so
existing downstream code (dq_reconciler, csv_tools, allocator, etc.) keeps
working unchanged.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from schemas import PlaybookExtraction
from tools.context_extractor import EXTRACTED_FILENAME, extract_and_persist


AUGMENTATIONS_FILENAME = "augmentations.json"


@dataclass(frozen=True)
class Playbook:
    """Bundle the deterministic pipeline reads from."""
    raw_markdown: str
    constants: Dict[str, Any]
    corridors: Dict[str, Any]
    item_master: List[Dict[str, Any]]
    aliases: List[Dict[str, Any]]
    legacy_map: List[Dict[str, Any]]
    uid_regex_rules: List[Dict[str, Any]]
    dq_config: Dict[str, Any]

    def corridor_ids(self) -> List[str]:
        return list(self.corridors.keys())

    def get_corridor(self, corridor_id: str) -> Dict[str, Any]:
        return self.corridors[corridor_id]


def _load_augmentations(augmented_dir: Path) -> Dict[str, Any]:
    path = augmented_dir / AUGMENTATIONS_FILENAME
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_corridors(extracted: PlaybookExtraction, augmentations: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ContextAgent's corridor data with the engineering augmentations.

    Output shape mirrors the legacy corridor_config.json so downstream code
    sees no difference.
    """
    aug_map = augmentations.get("corridor_augmentations", {})
    out: Dict[str, Any] = {}
    for c in extracted.corridors:
        aug = aug_map.get(c.corridor_id, {})
        if not aug:
            raise ValueError(
                f"Corridor '{c.corridor_id}' has no engineering augmentation in "
                f"augmentations.json. Add sla_tier, max_transit_hours, and "
                f"base_transit_hours to the corridor_augmentations block."
            )
        out[c.corridor_id] = {
            "corridor_name": c.corridor_name,
            "origin_dc": c.origin_dc,
            "destination_region": c.destination_region,
            "default_sla_tier_label": c.default_sla_tier_label,
            "sla_tier": aug["sla_tier"],
            "max_transit_hours": aug["max_transit_hours"],
            "base_transit_hours": aug["base_transit_hours"],
            "waypoints": [wp.model_dump() for wp in c.waypoints],
        }
    return out


def _build_constants(extracted: PlaybookExtraction, augmentations: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the constants dict that downstream code expects."""
    buffer_lookup = {
        str(row.risk_score): {"buffer_pct": row.buffer_pct, "escalation": row.escalation}
        for row in extracted.travel_buffer_by_risk
    }
    truck_aug = augmentations.get("truck_augmentations", {})
    return {
        "weather_thresholds": extracted.weather_thresholds.model_dump(),
        "travel_buffer_by_risk": buffer_lookup,
        "truck": {
            "capacity_volume_units": extracted.truck.capacity_volume_units,
            "packing_inefficiency_buffer_pct": extracted.truck.packing_inefficiency_buffer_pct,
            "effective_units_per_truck": truck_aug["effective_units_per_truck"],
        },
        "penalty_model": extracted.penalty_model.model_dump(),
        "sla_tiers": [t.model_dump() for t in extracted.sla_tiers],
        "planning_horizon_days": ["Day0", "Day1"],
    }


def load_playbook(
    playbook_md_path: str | Path,
    augmented_dir: str | Path,
    force_re_extract: bool = False,
) -> Playbook:
    """Run/load the ContextAgent extraction, merge augmentations, return Playbook.

    Args:
        playbook_md_path: path to the playbook markdown
        augmented_dir: directory containing augmentations.json and
            cached extracted_context.json
        force_re_extract: if True, re-invoke ContextAgent even if cached
    """
    augmented_dir = Path(augmented_dir)
    extraction = extract_and_persist(
        playbook_md_path=playbook_md_path,
        augmented_dir=augmented_dir,
        force=force_re_extract,
    )
    augmentations = _load_augmentations(augmented_dir)

    return Playbook(
        raw_markdown=Path(playbook_md_path).read_text(encoding="utf-8"),
        constants=_build_constants(extraction, augmentations),
        corridors=_merge_corridors(extraction, augmentations),
        item_master=[r.model_dump() for r in extraction.item_master],
        aliases=[a.model_dump() for a in extraction.aliases],
        legacy_map=[m.model_dump() for m in extraction.legacy_map],
        uid_regex_rules=[u.model_dump() for u in extraction.uid_regex_rules],
        dq_config=augmentations.get("data_quality_config", {}),
    )
