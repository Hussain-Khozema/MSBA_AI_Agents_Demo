"""Data Quality reconciliation per Appendix A.6 of the SeeWeeS playbook.

Pure-function module. Given a list of raw shipment rows and the playbook,
returns a ReconciliationResult containing one ReconciledRow per input row
with a stable reason_code, plus aggregate counts for downstream reporting.

Decision rules implemented (in precedence order):
    D1  Missing unique_item_id      -> excluded_missing_uid          (DQ-01)
    D3  Exact (item_id, item_name)  -> exact_match
    D4  Alias name match            -> alias_match
    D5  Legacy item_id match        -> legacy_id_map
    D6  Conflict / unresolved       -> excluded_unresolved           (DQ-02/03)
    D8  Duplicate unique_item_id    -> flagged_duplicate_uid         (DQ-04)

Notes:
    - A.5 regex-based unique_item_id regeneration is intentionally DISABLED
      for v1 (footgun risk; excluding 5 rows is cleaner to defend).
    - Substitution (A.4) is out of scope here; the allocator handles
      cold-chain truck assignment downstream.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter


REASON_EXACT_MATCH = "exact_match"
REASON_ALIAS_MATCH = "alias_match"
REASON_LEGACY_ID_MAP = "legacy_id_map"
REASON_GENERATED_IDENTIFIER = "generated_identifier"
REASON_EXCLUDED_MISSING_UID = "excluded_missing_uid"
REASON_EXCLUDED_UNRESOLVED = "excluded_unresolved"
REASON_FLAGGED_DUPLICATE_UID = "flagged_duplicate_uid"

DQ_RULE_MAP = {
    REASON_EXCLUDED_MISSING_UID: "DQ-01",
    REASON_EXCLUDED_UNRESOLVED: "DQ-02/03",
    REASON_FLAGGED_DUPLICATE_UID: "DQ-04",
    REASON_GENERATED_IDENTIFIER: "DQ-01-RESCUED-A5",
}


# Product class -> 3-letter prefix used in the generated placeholder UID.
# Derived from the example column of Appendix A.5.
PRODUCT_CLASS_PREFIX = {
    "Antiviral":         "RMD",
    "Oncology Biologic": "PMB",
    "Emergency":         "EPI",
    "Controlled":        "CTRL",
    "Respiratory":       "INH",
    "Clinical Trial":    "CT",
    "Endocrine":         "INS",
    "Anticoagulant":     "HEP",
}


@dataclass
class ReconciledRow:
    """One reconciled shipment unit with full provenance."""
    row_index: int
    original: Dict[str, Any]
    canonical_item_id: Optional[str]
    canonical_item_name: Optional[str]
    medicine_type: Optional[str]
    temp_control: Optional[str]
    product_class: Optional[str]
    is_cold_chain: Optional[bool]
    sla_tier: Optional[int]
    reason_code: str
    confidence_tier: Optional[str]
    dq_rule: Optional[str]
    is_valid_for_dispatch: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationResult:
    rows: List[ReconciledRow] = field(default_factory=list)

    @property
    def valid_rows(self) -> List[ReconciledRow]:
        return [r for r in self.rows if r.is_valid_for_dispatch]

    @property
    def excluded_rows(self) -> List[ReconciledRow]:
        return [r for r in self.rows if not r.is_valid_for_dispatch]

    def counts_by_reason(self) -> Dict[str, int]:
        return dict(Counter(r.reason_code for r in self.rows))

    def summary(self) -> Dict[str, Any]:
        by_reason = self.counts_by_reason()
        return {
            "total_rows": len(self.rows),
            "valid_for_dispatch": len(self.valid_rows),
            "excluded": len(self.excluded_rows),
            "flagged_for_investigation": sum(
                1 for r in self.rows if r.is_valid_for_dispatch and r.dq_rule
            ),
            "by_reason_code": by_reason,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    return " ".join(str(name).strip().lower().split())


def _build_indexes(
    item_master: List[Dict[str, Any]],
    aliases: List[Dict[str, Any]],
    legacy_map: List[Dict[str, Any]],
) -> Tuple[
    Dict[Tuple[int, str], Dict[str, Any]],  # exact (item_id, normalized_name) -> master row
    Dict[int, List[Dict[str, Any]]],         # item_id -> [master rows]  (for D6 conflict detection)
    Dict[str, Dict[str, Any]],               # normalized alias -> alias row
    Dict[int, Dict[str, Any]],               # legacy item_id -> legacy row
    Dict[str, Dict[str, Any]],               # canonical_item_id -> master row
]:
    exact: Dict[Tuple[int, str], Dict[str, Any]] = {}
    by_item_id: Dict[int, List[Dict[str, Any]]] = {}
    by_canonical: Dict[str, Dict[str, Any]] = {}
    for row in item_master:
        key = (int(row["item_id"]), _normalize_name(row["canonical_item_name"]))
        exact[key] = row
        by_item_id.setdefault(int(row["item_id"]), []).append(row)
        by_canonical[row["canonical_item_id"]] = row

    alias_idx: Dict[str, Dict[str, Any]] = {
        _normalize_name(a["alias_name"]): a for a in aliases
    }
    legacy_idx: Dict[int, Dict[str, Any]] = {
        int(m["legacy_item_id"]): m for m in legacy_map
    }
    return exact, by_item_id, alias_idx, legacy_idx, by_canonical


def _is_cold_chain(temp_control: Optional[str]) -> bool:
    """Anything not 'Room Temp (20-25C)' is cold-chain. Documented in playbook_constants.json."""
    if not temp_control:
        return False
    return "room temp" not in temp_control.lower()


# ---------------------------------------------------------------------------
# Core reconciliation
# ---------------------------------------------------------------------------

def _resolve_item(
    item_id_raw: Any,
    item_name: Any,
    exact_idx: Dict[Tuple[int, str], Dict[str, Any]],
    alias_idx: Dict[str, Dict[str, Any]],
    legacy_idx: Dict[int, Dict[str, Any]],
    by_canonical: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], str]:
    """Apply D3 -> D4 -> D5 precedence. Returns (canonical_row, reason, confidence, note)."""
    try:
        item_id = int(item_id_raw) if item_id_raw is not None else None
    except (TypeError, ValueError):
        item_id = None

    if item_id is not None:
        key = (item_id, _normalize_name(item_name))
        if key in exact_idx:
            row = exact_idx[key]
            return row, REASON_EXACT_MATCH, "EXACT_MATCH", f"Exact match on item_id={item_id} + name."

    norm = _normalize_name(item_name)
    if norm in alias_idx:
        alias_row = alias_idx[norm]
        row = by_canonical.get(alias_row["canonical_item_id"])
        if row is not None:
            return row, REASON_ALIAS_MATCH, alias_row.get("confidence_tier", "ALIAS_MATCH"), (
                f"Alias match: '{item_name}' -> {row['canonical_item_id']} ({alias_row.get('notes','')})."
            )

    if item_id is not None and item_id in legacy_idx:
        legacy_row = legacy_idx[item_id]
        row = by_canonical.get(legacy_row["canonical_item_id"])
        if row is not None:
            return row, REASON_LEGACY_ID_MAP, "LEGACY_ID_MAP", (
                f"Legacy item_id {item_id} mapped to {row['canonical_item_id']}: "
                f"{legacy_row.get('rationale','')}"
            )

    return None, None, None, ""


def _generate_uid(
    product_class: Optional[str],
    seq_counters: Dict[str, int],
    uid_regex_rules: List[Dict[str, Any]],
    format_template: str,
) -> Optional[str]:
    """Mint a placeholder unique_item_id keyed off the product_class.

    Uses PRODUCT_CLASS_PREFIX for the leading token and a per-class counter
    so the generated IDs are stable within a run and don't collide with
    each other. The format is deliberately distinct from real IDs (GEN###)
    so the audit trail can spot them at a glance.
    """
    if not product_class:
        return None
    prefix = PRODUCT_CLASS_PREFIX.get(product_class)
    if not prefix:
        # Try the regex example for a defensive fallback (e.g. take the
        # leading alpha token before the first dash from the example).
        for rule in uid_regex_rules:
            if rule.get("product_class") == product_class:
                example = rule.get("example", "")
                if example and "-" in example:
                    prefix = example.split("-", 1)[0]
                    break
    if not prefix:
        return None

    seq_counters[prefix] = seq_counters.get(prefix, 0) + 1
    return format_template.format(prefix=prefix, seq=seq_counters[prefix])


def reconcile_shipments(
    rows: List[Dict[str, Any]],
    item_master: List[Dict[str, Any]],
    aliases: List[Dict[str, Any]],
    legacy_map: List[Dict[str, Any]],
    corridors: Dict[str, Any],
    uid_regex_rules: Optional[List[Dict[str, Any]]] = None,
    dq_config: Optional[Dict[str, Any]] = None,
) -> ReconciliationResult:
    """Apply A.6 decision rules to every input row.

    When `dq_config["enable_uid_regeneration_a5"]` is True and a row has a
    missing `unique_item_id` but its item resolves successfully, we mint a
    placeholder UID per Appendix A.5 and mark the row reason='generated_identifier'
    instead of excluding it via DQ-01. Generated UIDs follow the format
    template in dq_config (e.g. 'RMD-2026-GEN001').

    Returns:
        ReconciliationResult with one ReconciledRow per input row.
    """
    exact_idx, by_item_id, alias_idx, legacy_idx, by_canonical = _build_indexes(
        item_master, aliases, legacy_map
    )
    uid_regex_rules = uid_regex_rules or []
    dq_config = dq_config or {}
    enable_regen = bool(dq_config.get("enable_uid_regeneration_a5", False))
    uid_format = dq_config.get("generated_uid_format", "{prefix}-2026-GEN{seq:03d}")

    # First pass: detect duplicate unique_item_id (DQ-04). Missing UIDs are
    # handled separately below and not counted as duplicates.
    uid_counts: Counter = Counter()
    for r in rows:
        uid = r.get("unique_item_id")
        if uid is not None and str(uid).strip() != "" and str(uid).lower() != "nan":
            uid_counts[str(uid).strip()] += 1
    duplicate_uids = {uid for uid, n in uid_counts.items() if n > 1}

    seq_counters: Dict[str, int] = {}
    result = ReconciliationResult()

    for idx, raw in enumerate(rows):
        uid = raw.get("unique_item_id")
        uid_str = str(uid).strip() if uid is not None else ""
        uid_missing = (uid_str == "" or uid_str.lower() == "nan")
        item_id_raw = raw.get("item_id")
        item_name = raw.get("item_name")
        corridor_id = raw.get("corridor_id")

        # Resolve the item first (D3 -> D4 -> D5). We need the product_class
        # before we can decide whether A.5 regeneration applies.
        canonical_row, reason, confidence, note = _resolve_item(
            item_id_raw, item_name, exact_idx, alias_idx, legacy_idx, by_canonical
        )

        # Unresolved item -> always exclude, regardless of UID state (D6)
        if canonical_row is None:
            result.rows.append(_make_excluded(
                idx, raw, REASON_EXCLUDED_UNRESOLVED,
                f"Unresolved item_id={item_id_raw} name='{item_name}' - no exact/alias/legacy match (DQ-02/03)."
            ))
            continue

        # Handle missing UID
        generated_uid: Optional[str] = None
        if uid_missing:
            if enable_regen:
                generated_uid = _generate_uid(
                    canonical_row.get("product_class"),
                    seq_counters, uid_regex_rules, uid_format,
                )
            if generated_uid is None:
                # Regeneration disabled or no prefix available -> DQ-01 exclude
                result.rows.append(_make_excluded(
                    idx, raw, REASON_EXCLUDED_MISSING_UID,
                    "Missing unique_item_id - excluded per DQ-01."
                ))
                continue
            uid_str = generated_uid
            reason = REASON_GENERATED_IDENTIFIER
            confidence = "GENERATED_A5"
            note = (
                f"Missing UID rescued via A.5: generated placeholder '{generated_uid}' "
                f"for product_class='{canonical_row.get('product_class')}'. "
                f"Resolved item via {('exact_match' if reason is None else reason)}."
            )

        # At this point the row has both a UID (real or generated) and a resolved item.
        sla_tier = corridors.get(corridor_id, {}).get("sla_tier") if corridor_id else None
        is_cold = _is_cold_chain(canonical_row.get("temp_control"))

        dq_rule = DQ_RULE_MAP.get(reason)
        if uid_str in duplicate_uids and not uid_missing:
            dq_rule = "DQ-04"
            note = note + f" | Duplicate unique_item_id '{uid_str}' - flagged for investigation."
            reason_to_store = REASON_FLAGGED_DUPLICATE_UID
        else:
            reason_to_store = reason  # type: ignore[assignment]

        result.rows.append(ReconciledRow(
            row_index=idx,
            original={**dict(raw), "unique_item_id": uid_str if not uid_missing else generated_uid},
            canonical_item_id=canonical_row["canonical_item_id"],
            canonical_item_name=canonical_row["canonical_item_name"],
            medicine_type=canonical_row.get("medicine_type"),
            temp_control=canonical_row.get("temp_control"),
            product_class=canonical_row.get("product_class"),
            is_cold_chain=is_cold,
            sla_tier=sla_tier,
            reason_code=reason_to_store,
            confidence_tier=confidence,
            dq_rule=dq_rule,
            is_valid_for_dispatch=True,
            notes=note,
        ))

    return result


def _make_excluded(idx: int, raw: Dict[str, Any], reason: str, note: str) -> ReconciledRow:
    return ReconciledRow(
        row_index=idx,
        original=dict(raw),
        canonical_item_id=None,
        canonical_item_name=None,
        medicine_type=None,
        temp_control=None,
        product_class=None,
        is_cold_chain=None,
        sla_tier=None,
        reason_code=reason,
        confidence_tier=None,
        dq_rule=DQ_RULE_MAP.get(reason),
        is_valid_for_dispatch=False,
        notes=note,
    )
