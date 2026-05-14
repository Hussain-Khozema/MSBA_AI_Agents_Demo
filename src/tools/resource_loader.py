"""Load the 48-hour resource availability CSV.

Schema: day, resource_type, available_count, notes
Resource types: driver, truck_standard, truck_temp_controlled
"""
from __future__ import annotations
from typing import Any, Dict

import pandas as pd


VALID_RESOURCE_TYPES = {"driver", "truck_standard", "truck_temp_controlled"}
VALID_DAYS = {"Day0", "Day1"}


def load_resource_availability(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Return {"Day0": {"driver": n, "truck_standard": n, "truck_temp_controlled": n, "_notes": {...}}, "Day1": {...}}."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    out: Dict[str, Dict[str, Any]] = {d: {"_notes": {}} for d in VALID_DAYS}
    for _, row in df.iterrows():
        day = str(row["day"]).strip()
        rtype = str(row["resource_type"]).strip()
        if day not in VALID_DAYS or rtype not in VALID_RESOURCE_TYPES:
            continue
        out[day][rtype] = int(row["available_count"])
        out[day]["_notes"][rtype] = str(row.get("notes", "")).strip()

    for day in VALID_DAYS:
        for rtype in VALID_RESOURCE_TYPES:
            out[day].setdefault(rtype, 0)
    return out
