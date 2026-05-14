# SeeWeeS Multi-Corridor Dispatch System

Multi-agent operations planner for time-critical medical shipments across
multiple delivery corridors. Built on **LangGraph** with three specialized
**LLM agents** (Context, Auditor, Report) wrapped around a deterministic
core (data-quality reconciliation, weather aggregation, penalty-minimizing
allocation).

This is a substantial rework of the starter repo for the **UCLA MSBA AI Agents
Project Challenge 2026**. We target enhancements **#5 (Multi-Region / Multi-Day
Resource Planning)** and **#1 (Self-Correction & Quality Assurance — audit
layer)**, plus a working implementation of **A.5 unique_item_id regeneration**.

---

## Table of Contents

1. [What changed vs. the starter repo](#1-what-changed-vs-the-starter-repo)
2. [Architecture diagram](#2-architecture-diagram)
3. [Project structure](#3-project-structure)
4. [Setup](#4-setup)
5. [Run the pipeline](#5-run-the-pipeline)
6. [Run the tests](#6-run-the-tests)
7. [Key assumptions](#7-key-assumptions)
8. [Agent design notes](#8-agent-design-notes)
9. [Validation: what the test suite covers](#9-validation-what-the-test-suite-covers)
10. [Limitations & next steps](#10-limitations--next-steps)

---

## 1. What changed vs. the starter repo


| Area                 | Starter                                           | This implementation                                                                 |
| -------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Graph topology       | Linear (6 nodes)                                  | Parallel fan-out + explicit join barrier (10 nodes)                                 |
| Number of LLM agents | 4 (Context, Ops, Planner, Report) — all narrative | 3 (Context, Auditor, Report) — all schema-enforced                                  |
| Playbook ingestion   | PDF + Chroma RAG                                  | Markdown + ContextAgent w/ pydantic structured outputs                              |
| Data-quality logic   | Generic IsolationForest anomaly hunt              | Full Appendix A.6 D1..D8 decision rules + A.5 UID regeneration                      |
| Corridors            | Single (Boston-only CSV)                          | Two (`C1_I95_NJ_BOS`, `C2_NJ_PHL`) with multi-waypoint weather                      |
| Weather risk         | Single point, single score                        | Per-waypoint, max-aggregated to corridor/day, then 48h                              |
| Allocation           | Done by LLM (narrative)                           | Deterministic Python greedy minimizing penalty score (Sec 13)                       |
| Audit / QA           | None                                              | 7 deterministic checks + LLM narrative wrapper (AuditorAgent)                       |
| Report               | Generic HTML, hand-waved metrics                  | Strict no-hallucination guardrails, every number precomputed, CSS skeleton injected |
| Tests                | One placeholder smoke test                        | 24 deterministic tests (0.4s runtime, zero LLM cost)                                |


---

## 2. Architecture diagram



Source: `[docs/architecture.puml](docs/architecture.puml)` (rendered PNG at
`[docs/architecture.png](docs/architecture.png)`).

The pipeline is **not linear**. After ContextAgent extracts the playbook,
three deterministic tools (ingest, weather, resources) run in **parallel**,
converge at an explicit `join` barrier, and then flow sequentially through
KPI computation, allocation, audit, and report. The three LLM agents are
highlighted in the diagram (green = Context, pink = Auditor, blue = Report);
everything else is deterministic Python.

---

## 3. Project structure

```
.
├── data/
│   └── augmented/
│       ├── augmentations.json           # hand-curated engineering decisions
│       └── extracted_context.json       # produced by ContextAgent (cached)
├── data-for-enhancement/
│   ├── SeeWeeS Specialty Dispatch Playbook.md
│   ├── Incoming_shipments_14d_multi_corridor.csv
│   └── Resource_availability_48h.csv
├── docs/
│   └── architecture.puml                # PlantUML source for the diagram
├── output/
│   └── report.html                      # generated executive report
├── src/
│   ├── main.py                          # entry point
│   ├── graph.py                         # LangGraph topology
│   ├── agents.py                        # ContextAgent / AuditorAgent / ReportAgent
│   ├── prompts.py                       # all three prompts
│   ├── schemas.py                       # pydantic schemas (PlaybookExtraction, AuditReport)
│   ├── tracing.py                       # LangSmith opt-in
│   └── tools/
│       ├── context_extractor.py         # ContextAgent orchestration + caching
│       ├── playbook_loader.py           # merges extraction + augmentations
│       ├── csv_tools.py                 # shipment CSV ingest + history/planning split
│       ├── dq_reconciler.py             # Appendix A.6 D1..D8 + A.5 regeneration
│       ├── weather_tools.py             # per-waypoint Open-Meteo, two-level aggregation
│       ├── resource_loader.py           # 48h resource availability
│       ├── kpi_engine.py                # corridor KPIs + period-over-period
│       ├── allocator.py                 # deterministic greedy solver
│       ├── auditor.py                   # 7 deterministic compliance checks
│       └── email_tools.py               # SMTP delivery
├── tests/
│   └── test_pipeline.py                 # 24 deterministic tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Setup

Requirements: Python 3.11+, an OpenAI API key.

```bash
# 1. Clone and enter the project
cd MSBA_AI_Agents_Demo

# 2. Create + activate a venv
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...
```

### Required environment variables


| Variable                                               | Required | Purpose                              |
| ------------------------------------------------------ | -------- | ------------------------------------ |
| `OPENAI_API_KEY`                                       | yes      | All three agents                     |
| `WEATHER_TZ`                                           | no       | Defaults to `America/New_York`       |
| `LANGCHAIN_TRACING_V2`                                 | no       | Set to `true` to enable LangSmith    |
| `LANGCHAIN_API_KEY`                                    | no       | Required if LangSmith tracing is on  |
| `REPORT_EMAIL_TO`                                      | no       | Enables SMTP delivery if set         |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | no       | Required if `REPORT_EMAIL_TO` is set |


---

## 5. Run the pipeline

```bash
python src/main.py
```

What this does end-to-end:

1. Invokes **ContextAgent** on the playbook markdown (cached after first run).
2. Merges the LLM extraction with `data/augmented/augmentations.json`.
3. Runs `ingest`, `weather`, `resources` in parallel:
  - Ingest: reconciles the 14-day shipment CSV against Appendix A.6 rules.
   With A.5 regeneration enabled (default), the 5 rows with missing
   `unique_item_id` are rescued with placeholder UIDs (e.g. `RMD-2026-GEN001`).
  - Weather: fetches 9 waypoint forecasts from Open-Meteo in parallel,
  aggregates waypoint → day → 48h corridor risk score.
  - Resources: loads driver / truck / reefer availability for Day0 & Day1.
4. Computes per-corridor KPIs + period-over-period trends.
5. Runs the deterministic greedy allocator minimizing the playbook's penalty
  model (Tier-1 SLA = 100, Tier-2 SLA = 40, cold-chain breach = +80,
   day-late = 10).
6. Runs **AuditorAgent**: 7 deterministic checks + LLM narrative.
7. Runs **ReportAgent**: renders the HTML report under `output/report.html`.
8. Optionally emails the report via SMTP.

Open the result:

```bash
open output/report.html
```

### Force a fresh ContextAgent extraction

The extraction is cached at `data/augmented/extracted_context.json` so
subsequent runs skip the LLM call. To force re-extraction (e.g. after the
playbook changes):

```bash
rm data/augmented/extracted_context.json
python src/main.py
```

---

## 6. Run the tests

```bash
pytest tests/ -v
```

Expected output: **24 tests pass in ~0.5s**. None of the tests touch the
LLM or external APIs — they exercise the deterministic core using the
cached extraction and a stubbed weather report. See
[section 9](#9-validation-what-the-test-suite-covers) for what each test
verifies.

---

## 7. Key assumptions

We made the following augmentations to the playbook. All are documented in
`data/augmented/augmentations.json` with a `rationale` field, and the test
suite verifies the system uses them correctly.

### SLA tier per corridor (mapping decision)

The playbook (§7) ties SLA tier to *medicine category*. For v1 we simplified
to a per-corridor mapping because the planning-window CSV is segregated by
destination region:


| Corridor        | SLA Tier | Max Transit | Base Transit (assumed) |
| --------------- | -------- | ----------- | ---------------------- |
| `C1_I95_NJ_BOS` | Tier 1   | 6h          | 5.0h                   |
| `C2_NJ_PHL`     | Tier 2   | 12h         | 2.0h                   |


`base_transit_hours` is estimated from highway distance at average truck speed.

### Cold-chain rule

Any item whose `temp_control` field in Appendix A.1 is **not** `"Room Temp (20-25C)"` is treated as cold-chain and requires a `truck_temp_controlled`.

### Effective truck capacity

The playbook §8 specifies 10 volume units capacity and a +10% packing
inefficiency buffer. We compute `effective_units_per_truck = floor(10 / 1.10) = 9`
to stay conservative.

### A.5 UID regeneration

The playbook permits regex-based regeneration of missing `unique_item_id`
values (Appendix A.5 + A.6 D1). We enable this via
`augmentations.json -> data_quality_config.enable_uid_regeneration_a5 = true`.
Generated UIDs follow a **clearly-marked placeholder format**
(e.g. `RMD-2026-GEN001`, `INS-2026-GEN001`) so they can never be confused
with real inventory IDs. Every generated UID is surfaced in the report's
"Audit Trail: Generated Identifier Rows" appendix.

Flip the flag to `false` to revert to strict DQ-01 exclusion.

---

## 8. Agent design notes

Three LLM agents, all wrapped with strict guardrails:

### ContextAgent

- Purpose: extract the SeeWeeS Specialty Dispatch Playbook into a structured
`PlaybookExtraction` object.
- **Schema-enforced** via `with_structured_output(PlaybookExtraction, method="json_schema")`
— the LLM literally cannot return malformed data.
- The prompt explicitly tells the agent what *not* to extract (engineering
augmentations, substitution policy) so it cannot hallucinate values that
are not in the markdown.
- Output cached to `data/augmented/extracted_context.json` (audit trail).

### AuditorAgent 

- Purpose: produce an executive-friendly `AuditReport`.
- **Hybrid design**: the deterministic Python core (`tools/auditor.py`) runs
7 compliance checks and produces pass/warn/fail with locked statuses.
The LLM only rewrites the `description` and `recommendation` text — it
**cannot change** any check_id, status, or severity.
- This guarantees the audit verdict is reproducible while keeping the prose
human-friendly.

### ReportAgent 

- Purpose: render the final HTML report.
- **Strict no-hallucination prompt**: "Every numeric value you mention MUST  
come from the JSON payload. Do not recompute totals. If a number is not  
in the payload, write '(not available)'."

---

## 9. Validation: what the test suite covers

All tests live in `tests/test_pipeline.py`. They run in **~0.5 seconds** and  
make **zero LLM calls** by using the cached ContextAgent extraction. Run with  
`pytest tests/ -v`.

### Coverage breakdown


| Area              | Test class               | What it verifies                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ContextAgent      | `TestPlaybookExtraction` | All 11 canonical items, 7 aliases, 4 legacy mappings, 6 UID regex rules are extracted; penalty model values are exact; corridor augmentations are applied.                                                                                                                                                                                                                        |
| Data quality      | `TestDqReconciler`       | Planning window is fully dispatchable with A.5 on (33/33); legacy IDs (`1070`, `99999`) resolve to canonical IDs; alias whitespace variants (`Remdesivir 100 mg`) resolve correctly; strength variants (`RMD-100` vs `RMD-200` sharing `item_id=10021`) are disambiguated by name; generated UIDs follow `*-GEN###` format; with A.5 off, the same 3 rows are excluded via DQ-01. |
| Allocator         | `TestAllocator`          | All 33 units served; capacity never exceeded on any day; every served cold-chain unit lands on `truck_temp_controlled`; C1's weather-driven Tier-1 SLA penalty is applied (not the unserved variant).                                                                                                                                                                             |
| AuditorAgent core | `TestAuditor`            | All 7 deterministic checks execute; escalation passes when max risk = 2 and *fails* when forced to 3; weather SLA breach correctly flags C1; cold-chain integrity passes; capacity math passes; DQ-regeneration produces a warn-level finding when 3 UIDs are generated.                                                                                                          |


### Sample real-run output (with live LLM + Open-Meteo)

```
Full report saved to: output/report.html
Total penalty: 2840
Summary: {'units_total': 33, 'units_served': 33, 'units_unserved': 0,
          'units_bumped': 0, 'tier1_unserved': 0, 'cold_chain_unserved': 0,
          'weather_sla_violated_corridors': ['C1_I95_NJ_BOS']}
```

The headline business signal the system surfaces:

- **All 33 valid planning units served** (3 rescued via A.5).
- **Capacity is not the bottleneck.**
- **Weather is**: C1's I-95 corridor hits risk-score 2 on Day 1 (worst at
Bronx NY / Providence RI), triggering a +25% buffer that pushes
transit to 6.25h vs the 6h Tier-1 cap.
- **Both corridors are running hot**: C1 is +104% above its 12-day
baseline, C2 is +78%.

The audit section reports **5 passed / 1 warned / 1 failed of 7 rules
evaluated**, with the failure being the C1 weather SLA breach and the
warning being the A.5 UID regeneration usage.

---

## 10. Limitations & next steps

- **Greedy allocator only.** An ILP version (e.g. `pulp` or `scipy.milp`)
would be tractable here (33 units × 2 days) and could show "Plan A
(greedy): 2840 penalty vs. Plan B (ILP): X penalty, chose B" — a stronger
story for the report. Out of scope for v1.
- **No cyclic audit loop.** The current audit is one-shot: it adds findings
to the report rather than feeding violations back into a re-allocation
step. The structural failures we observe (weather-induced SLA breach)
cannot be fixed by re-allocation alone — they require either earlier
dispatch (out of horizon) or alternate transport mode — so we surface
them in the audit section as actionable findings instead.
- **Human-in-the-loop.** A `human_checkpoint` interrupt could be added when
`max_48h_risk_score == 3` or when `tier1_unserved > 0`. Would require
LangGraph `interrupt_before` and a manager-facing CLI.
- `**base_transit_hours` is a static estimate.** A production version would
pull live ETA from a routing API and account for traffic.
- **Penalty weights are taken verbatim from the playbook.** A real ops team
would tune these to match historical SLA-violation costs.
- **No live retraining.** The DQ reconciler doesn't learn new alias patterns
from operator feedback; new aliases must be added to the playbook
appendix and re-extracted.

