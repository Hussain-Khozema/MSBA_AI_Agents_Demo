# data-for-enhancement/ — Student Artifacts for Enhancements

This folder contains **student-facing artifacts** designed for enhancement challenges. The key idea is:

- Shipment CSVs provide the **raw operational feed** (with intentional data-quality issues).
- The playbook provides the **rules of the business** and the **authoritative appendix tables** to resolve certain issues.

---

## What’s in this folder

### Shipment CSVs
- **`Incoming_shipments_14d_multi_corridor.csv`**
  - A 14-day shipment feed spanning **two corridors** used for trend analysis, data-quality reconciliation, multi-region planning and resource allocation.
  - **Schema** (one physical unit per row):
    - `shipment_date`
    - `planning_day`
    - `is_planning_window`
    - `corridor_id` 
    - `item_id`
    - `item_name`
    - `unique_item_id`
    - `dispatch_location`
  - Intentionally includes issues like missing identifiers and legacy IDs.

### Playbook (rules + appendix tables)
- **`SeeWeeS Specialty Dispatch Playbook.md`**
  - Operational rules (weather risk, capacity model, reporting requirements)
  - Data quality rules (DQ-01..DQ-04)
  - **Appendix A — Item Master Appendix (Searchable)**: the main reference for reconciling identifiers and standardizing analytics keys.
  - **Multi-corridor + 48-hour planning policy** (corridors, waypoints, aggregation, and resource allocation objective)

### Resource constraints (for allocation problems)
- **`Resource_availability_48h.csv`**
  - Daily availability of scarce resources (`driver`, `truck_standard`, `truck_temp_controlled`) used by the planner.

---

## How students should use these artifacts (quick guide)

- Use the playbook’s **Corridor Catalog** to interpret corridor geography, waypoints, and default SLAs for each corridor.
- Use `Incoming_shipments_14d_multi_corridor.csv` to:
  - slice shipments by **corridor** and **day** or for period-over-period analysis,
  - compute corridor-level KPIs (volume, mix, risk) over the 48h planning window,
  - compare how workload and risk differ across corridors.
- Use `Resource_availability_48h.csv` to:
  - understand daily availability of `driver`, `truck_standard`, and `truck_temp_controlled`,
  - propose an allocation of scarce resources across corridors and days that maximizes impact while respecting constraints.
- Use Appendix A (Item Master) to standardize item identifiers before computing KPI comparisons (strongly recommended for cleaner analytics).
- When writing up recommendations, explicitly state:
  - which corridors receive more/less capacity and why,
  - any remaining backlog or risk and how you would monitor it over the 48h horizon.
