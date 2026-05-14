"""Shipment CSV ingestion + reconciliation.

Loads the 14-day multi-corridor shipment feed, splits History vs Planning
Window, runs the DQ reconciler over both slices, and exposes per-corridor
per-day groupings consumable by the KPI engine and the allocator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from tools.dq_reconciler import (
    ReconciliationResult,
    ReconciledRow,
    reconcile_shipments,
)
from tools.playbook_loader import Playbook


REQUIRED_COLUMNS = {
    "shipment_date",
    "planning_day",
    "is_planning_window",
    "corridor_id",
    "item_id",
    "item_name",
    "unique_item_id",
    "dispatch_location",
}


@dataclass
class ShipmentDataset:
    """Output of the ingestion stage."""
    history: ReconciliationResult
    planning: ReconciliationResult
    raw_total_rows: int
    raw_history_rows: int
    raw_planning_rows: int

    def planning_valid_by_corridor_day(self) -> Dict[str, Dict[str, List[ReconciledRow]]]:
        """Return {corridor_id: {planning_day: [valid rows]}} for the allocator."""
        out: Dict[str, Dict[str, List[ReconciledRow]]] = {}
        for r in self.planning.valid_rows:
            cid = r.original.get("corridor_id")
            day = r.original.get("planning_day")
            if cid is None or day is None:
                continue
            out.setdefault(cid, {}).setdefault(day, []).append(r)
        return out


def _df_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert df to row dicts with NaN -> None for clean reconciler input."""
    return df.where(pd.notna(df), None).to_dict(orient="records")


def load_and_reconcile_shipments(csv_path: str, playbook: Playbook) -> ShipmentDataset:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")

    history_df = df[df["is_planning_window"] == 0].copy()
    planning_df = df[df["is_planning_window"] == 1].copy()

    history_res = reconcile_shipments(
        _df_to_rows(history_df),
        playbook.item_master, playbook.aliases, playbook.legacy_map, playbook.corridors,
        uid_regex_rules=playbook.uid_regex_rules,
        dq_config=playbook.dq_config,
    )
    planning_res = reconcile_shipments(
        _df_to_rows(planning_df),
        playbook.item_master, playbook.aliases, playbook.legacy_map, playbook.corridors,
        uid_regex_rules=playbook.uid_regex_rules,
        dq_config=playbook.dq_config,
    )

    return ShipmentDataset(
        history=history_res,
        planning=planning_res,
        raw_total_rows=len(df),
        raw_history_rows=len(history_df),
        raw_planning_rows=len(planning_df),
    )
