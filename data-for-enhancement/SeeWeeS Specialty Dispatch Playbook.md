# SeeWeeS Specialty Dispatch Playbook
**Multi-Corridor Dispatch Planning (48-hour Horizon)**  
**Version 0.2 — Internal Operations Reference**

This document is designed to be machine-readable by automated planning agents.

## 1. Purpose & Scope
- Defines dispatch planning rules, weather risk logic, and data quality standards.
- Applies to time-critical medicine shipments originating from New Jersey distribution centers.
- Destination: multiple hospital regions across defined delivery corridors.
- Planning horizon: next **48 hours** (Day0 + Day1).

## 2. System Overview (Narrative Context)
- Medicines are shipped daily from NJ distribution centers across multiple delivery corridors.
- Delivery is time-critical (patient care, cold-chain constraints).
- Dispatch decisions must account for:
  - Weather-driven travel risk
  - Item master data accuracy
  - Truck capacity and packing constraints

## 3. Corridor Catalog (Authoritative)
The system supports multiple delivery corridors. Corridors are the unit of comparison for risk, KPIs, and resource allocation.

### 3.1 Corridors
| corridor_id | corridor_name | origin_dc | destination_region | default_sla_tier | notes |
|---|---|---|---|---|---|
| C1_I95_NJ_BOS | NJ → Boston (I-95) | Newark_NJ_DC | Boston_MA | Tier 1 | Existing corridor |
| C2_NJ_PHL | NJ → Philadelphia | Newark_NJ_DC | Philadelphia_PA | Tier 2 | Added corridor for multi-region planning |

### 3.2 Waypoints (per corridor)
Weather risk is evaluated independently at each waypoint. Corridor risk is computed from waypoint risk values across the planning horizon.

**C1_I95_NJ_BOS** (existing)
| waypoint_id | city | lat | lon |
|---|---|---:|---:|
| C1_W1 | Newark NJ | 40.7357 | -74.1724 |
| C1_W2 | Bronx NY | 40.8448 | -73.8648 |
| C1_W3 | New Haven CT | 41.3083 | -72.9279 |
| C1_W4 | Providence RI | 41.8240 | -71.4128 |
| C1_W5 | Boston MA | 42.3601 | -71.0589 |

**C2_NJ_PHL** (new)
| waypoint_id | city | lat | lon |
|---|---|---:|---:|
| C2_W1 | Newark NJ | 40.7357 | -74.1724 |
| C2_W2 | New Brunswick NJ | 40.4862 | -74.4518 |
| C2_W3 | Trenton NJ | 40.2204 | -74.7643 |
| C2_W4 | Philadelphia PA | 39.9526 | -75.1652 |

## 4. Authoritative Data Sources

### 3.1 Weather
- Source: Open-Meteo Weather Forecast API
- Granularity: Daily aggregates
- Forecast window: Next 2 forecast days (`forecast_days = 2`) to cover “next-day” planning
- Variables used:
  - `precipitation_sum` (mm/day)
  - `wind_gusts_10m_max` (km/h)
  - `temperature_2m_min` (°C)
- Hourly variables retrieved (optional reporting only; not used in risk score):
  - `temperature_2m`
  - `precipitation`
  - `wind_speed_10m`
  - `wind_gusts_10m`

### 3.2 Item Master Data
Source of truth for:
- `item_id`
- `item_name`
- medicine type
- packaging volume
- temperature requirements

## 5. Planning Horizon (48 hours) and Aggregation
The planning window covers **Day0** and **Day1** (48 hours). Weather is retrieved with `forecast_days = 2`.

### 5.1 Corridor risk aggregation across days (default)
Compute a corridor/day risk score first (0–3), then aggregate to a 48-hour corridor risk.

- **Day risk**: apply the waypoint trigger logic to each waypoint for that day; corridor day risk is the **max** waypoint score.
- **48-hour corridor risk**: take the **max** of Day0 and Day1 corridor day risk.

Teams may propose an alternative aggregation (e.g., weighted max) but must justify it in their report.

## 6. Weather Risk Rules (Operational Logic)

### 5.1 Weather Triggers (Daily Index)
Weather risk is triggered if any waypoint meets one or more conditions across the next 2 forecast days:

| Condition | Open-Meteo Daily Variable | Threshold |
|---|---|---|
| Heavy Precipitation Risk | `precipitation_sum` | ≥ 15.0 mm/day |
| High Wind Risk | `wind_gusts_10m_max` | ≥ 45.0 km/h |
| Freezing Risk | `temperature_2m_min` | ≤ 0.0 °C |

### 5.2 Travel Time Buffer Policy (Score-based)
The `PlannerAgent` converts `risk_score_0_3` into a travel-time buffer:

| risk_score_0_3 | Travel Time Adjustment |
|---:|---|
| 0 | No buffer |
| 1 | +10% buffer |
| 2 | +25% buffer |
| 3 | +40% buffer + escalation |

## 7. Dispatch SLA Classes
| SLA Tier | Medicine Category | Max Time-in-Transit |
|---|---|---|
| Tier 1 | Life-critical | 6 hours |
| Tier 2 | Standard specialty | 12 hours |

Dispatch plans violating SLA must be flagged.

## 8. Truck Capacity & Packing Model

### 7.1 Truck Constraints
- Standard truck capacity: **10 volume units**
- Each `unique_item_id` is **1 volume unit**
- Packing inefficiency buffer: **+10%**
- Cold-chain items require **temperature-controlled trucks**

### 7.2 Volume Calculation
- \( \text{total\_volume} = \sum(\text{volume\_per\_unique\_item\_id}) \)
- \( \text{required\_trucks} = \lceil (\text{total\_volume} \times 1.10) / \text{truck\_capacity} \rceil \)

## 9. Medicine Master Reference (Item Truth Table)
This table is the authoritative mapping used for validation.

| item_id | item_name | medicine_type | temp_control |
|---:|---|---|---|
| 10021 | Remdesivir 100mg | Antiviral | Cold (2-8C) |
| 10021 | Remdesivir 200mg | Antiviral | Cold (2-8C) |
| 10022 | Insulin Lispro | Hormone | Cold (2-8C) |
| 10035 | Pembrolizumab | Monoclonal Antibody | Cold (2-8C) |
| 10040 | Epinephrine Auto-Injector | Emergency Drug | Room Temp (20-25C) |
| 10050 | Heparin Sodium | Anticoagulant | Room Temp (20-25C) |
| 10060 | Morphine Sulfate | Opioid Analgesic | Controlled Storage |
| 10070 | Albuterol Inhaler | Bronchodilator | Room Temp (20-25C) |
| 99999 | Experimental Oncology Drug | Clinical Trial Drug | Strict Cold Chain (-20C) |

## 10. Shipment CSV Schema
Incoming shipment files must contain:

| Column Name | Description |
|---|---|
| `shipment_date` | Dispatch day (used for Day0/Day1 comparisons) |
| `item_id` | Internal medicine identifier |
| `item_name` | Human-readable name |
| `unique_item_id` | Unit-level identifier |
| `dispatch_location` | Dispatch Hospital |
| `corridor_id` | Delivery corridor key (required for multi-corridor planning) |

Each row represents one physical unit.

## 11. Data Quality Rules (Anomaly Definitions)
| Rule ID | Description | Action |
|---|---|---|
| DQ-01 | Missing `unique_item_id` | Remove from the dispatch calculation |
| DQ-02 | `item_id` not in master table | Flag for investigation |
| DQ-03 | `item_name` mismatch for valid `item_id` | Flag for investigation |
| DQ-04 | Duplicate `unique_item_id` | Flag for investigation |

## 12. Exception Handling Policy
- Invalid rows are excluded from dispatch planning.
- All excluded rows must be:
  - Counted
  - Logged
  - Reported with reason codes

## 13. Resource Constraints and Allocation Policy (New)
Planning must account for limited resources shared across corridors and days.

### 13.1 Resource pools
Treat the following as scarce pools by day:
- `driver`
- `truck_standard`
- `truck_temp_controlled`

If a separate resource file is provided, that file is authoritative for daily availability.

### 13.2 Allocation objective (measurable)
When resources are insufficient, allocate to **minimize total penalty score** across corridors and days.

Use the following penalty model (per affected shipment unit):

| Violation Type | Penalty per Unit |
|---|---:|
| Tier 1 SLA violation | 100 points |
| Tier 2 SLA violation | 40 points |
| Cold-chain (temp-control) violation (any tier) | +80 points (in addition to SLA penalty, if any) |
| Non-SLA delivery delay (still within SLA, but not dispatched on requested day) | 10 points |

**Total penalty score** is:
- Sum all penalties for all impacted units across both days and all corridors.

The **allocation objective** is therefore:
1) Choose a dispatch plan that yields the **lowest total penalty score**.
2) If two plans tie on total penalty, prefer the plan with **fewer Tier 1 units impacted**.

This quantitative model makes corridor-level trade-offs explicit and gradable. Teams may propose adjustments, but any alternative must:
- Be defined with explicit numeric penalties/weights.
- Be applied consistently across corridors and days.
- Be documented in the final report (including rationale and example calculations).

## 14. Reporting Requirements
Final dispatch report must include:
- Weather risk summary (by corridor, and by day within the 48h window)
- Applied travel buffers
- Valid vs excluded shipment counts
- KPI comparison across corridors (shipment volume, excluded rate, Tier 1/Tier 2 mix)
- Recommended resource allocation (drivers/trucks/temp-controlled) by corridor and day, with rationale
- SLA risk flags

---

## Appendix A — Item Master Appendix (Searchable)
This appendix is the operational reference for resolving missing/invalid identifiers and for standardizing analytics keys across item master variants.

### A.1 Canonical Item Master (Authoritative)
Canonical keys are used for analytics and reconciliation. The shipment field `item_id` may not be globally unique (e.g., strength variants); use `canonical_item_id` as the stable key.

| canonical_item_id | item_id | canonical_item_name | medicine_type | temp_control | product_class |
|---|---:|---|---|---|---|
| RMD-100 | 10021 | Remdesivir 100mg | Antiviral | Cold (2-8C) | Antiviral |
| RMD-200 | 10021 | Remdesivir 200mg | Antiviral | Cold (2-8C) | Antiviral |
| INS-LIS | 10022 | Insulin Lispro | Hormone | Cold (2-8C) | Endocrine |
| PMB-KEY | 10035 | Pembrolizumab | Monoclonal Antibody | Cold (2-8C) | Oncology Biologic |
| EPI-AI | 10040 | Epinephrine Auto-Injector | Emergency Drug | Room Temp (20-25C) | Emergency |
| HEP-SOD | 10050 | Heparin Sodium | Anticoagulant | Room Temp (20-25C) | Anticoagulant |
| MOR-SUL | 10060 | Morphine Sulfate | Opioid Analgesic | Controlled Storage | Controlled |
| ALB-INH | 10070 | Albuterol Inhaler | Bronchodilator | Room Temp (20-25C) | Respiratory |
| EXP-ONC-CT | 99999 | Experimental Oncology Drug (Clinical Trial) | Clinical Trial Drug | Strict Cold Chain (-20C) | Clinical Trial |
| LEV-INH | 10071 | Levalbuterol Inhaler | Bronchodilator | Room Temp (20-25C) | Respiratory |
| INS-ASP | 10023 | Insulin Aspart | Hormone | Cold (2-8C) | Endocrine |

### A.2 Name Alias / Variant Table (Accepted)
Use this table when `item_name` contains typos/variants. Alias matches create a mapping suggestion with a confidence tier.

| alias_name | canonical_item_id | confidence_tier | notes |
|---|---|---|---|
| Remdesivir 100 mg | RMD-100 | ALIAS_MATCH | Space variant |
| Remdesivir 200 mg | RMD-200 | ALIAS_MATCH | Space variant |
| Pembrolizumab (Keytruda) | PMB-KEY | ALIAS_MATCH | Brand in parentheses |
| EpiPen Auto Injector | EPI-AI | ALIAS_MATCH | Common brand phrasing |
| Heparin Na | HEP-SOD | ALIAS_MATCH | Abbreviation |
| Morphine Sulphate | MOR-SUL | ALIAS_MATCH | US/UK spelling |
| Albuterol Inhaler 90mcg | ALB-INH | ALIAS_MATCH | Dose suffix ignored for inhaler class |

### A.3 Legacy / Deprecated / Invalid Identifier Mapping
Use this table when `item_id` is unknown/invalid in the master but can be reconciled to a canonical item. All legacy-mapped rows must be reported as “fixed via legacy mapping.”

| legacy_item_id | canonical_item_id | rule | rationale |
|---:|---|---|---|
| 10020 | RMD-100 | LEGACY_ID_MAP | Vendor legacy ID for Remdesivir 100mg |
| 20021 | RMD-200 | LEGACY_ID_MAP | Old system used 200xx for strength variants |
| 1070 | ALB-INH | LEGACY_ID_MAP | Truncated ID found in older CSV exports |
| 99999 | EXP-ONC-CT | SPECIAL_CASE | Clinical trial placeholder ID; requires strict cold chain |

### A.4 Substitution Policy (Operationally Allowed Cases)
Substitution is only allowed when explicitly permitted below. If not listed, do not substitute—flag for investigation.

| canonical_item_id | substitute_canonical_item_id | allowed_when | not_allowed_when |
|---|---|---|---|
| ALB-INH | LEV-INH | Same medicine_type=Bronchodilator AND temp_control matches AND hospital approves interchange | Patient-specific order forbids substitution or formulary restriction applies |
| INS-LIS | INS-ASP | Cold-chain endocrine hormones only; substitution requires pharmacist approval | Patient-specific shipment or dosing regimen differs |

### A.5 Identifier Format Rules (`unique_item_id`)
These patterns help validate and (when safe) regenerate missing `unique_item_id` values. If regeneration is used, mark as `generated_identifier` and log the basis.

| product_class | expected_regex | example | notes |
|---|---|---|---|
| Antiviral | `^RMD-\d{4}-\d{4}$` | RMD-2026-0042 | Year + 4-digit sequence |
| Oncology Biologic | `^PMB-\d{4}-\d{5}$` | PMB-2026-00017 | Year + 5-digit sequence |
| Emergency | `^EPI-\d{4}-\d{4}$` | EPI-2026-0101 | Year + 4-digit sequence |
| Controlled | `^CTRL-\d{4}-\d{6}$` | CTRL-2026-000123 | Year + 6-digit sequence |
| Respiratory | `^INH-\d{4}-\d{5}$` | INH-2026-00234 | Year + 5-digit sequence |
| Clinical Trial | `^CT-\d{4}-[A-Z0-9]{6}$` | CT-2026-A1B2C3 | Randomized suffix |

### A.6 Decision Rules (How to Use This Appendix)
- **D1 Precedence**: If `unique_item_id` is missing, follow DQ-01 unless an explicit regeneration rule (A.5) is permitted and logged.
- **D2 Canonical key**: Reconcile to `canonical_item_id` for analytics; preserve raw fields in logs when possible.
- **D3 Exact match**: If (`item_id`, `item_name`) maps to a single canonical row in A.1, confidence = `EXACT_MATCH`.
- **D4 Alias match**: If `item_name` matches A.2, map to `canonical_item_id`, confidence = `ALIAS_MATCH`.
- **D5 Legacy match**: If `item_id` matches A.3, map to `canonical_item_id`, confidence = `LEGACY_ID_MAP`.
- **D6 Conflicts**: If `item_id` points to multiple canonical items, use `item_name` to disambiguate; otherwise flag `UNRESOLVED_CONFLICT`.
- **D7 Substitution**: Only substitute when listed in A.4 and approval condition is met; otherwise flag (no auto-fix).
- **D8 Reporting**: Every modified row must include a reason code: `exact_match`, `alias_match`, `legacy_id_map`, `substituted`, `generated_identifier`, `excluded_unresolved`.

