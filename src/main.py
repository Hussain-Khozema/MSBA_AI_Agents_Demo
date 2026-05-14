"""Entry point for the SeeWeeS multi-corridor dispatch pipeline (v2).

Reads:
    - data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md
    - data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv
    - data-for-enhancement/Resource_availability_48h.csv
    - data/augmented/*.json

Writes:
    - output/report.html
"""
from __future__ import annotations
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # before tracing/agent imports

from tracing import init_langsmith_tracing
init_langsmith_tracing()

from graph import build_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data-for-enhancement"
AUG_DIR = PROJECT_ROOT / "data" / "augmented"


def _strip_html_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s


HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SeeWeeS Multi-Corridor Dispatch Report</title>
</head>
<body>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    app = build_graph()

    initial_state = {
        "playbook_md_path":   str(DATA_DIR / "SeeWeeS Specialty Dispatch Playbook.md"),
        "augmented_dir":      str(AUG_DIR),
        "shipments_csv_path": str(DATA_DIR / "Incoming_shipments_14d_multi_corridor.csv"),
        "resources_csv_path": str(DATA_DIR / "Resource_availability_48h.csv"),
    }

    final = app.invoke(initial_state)

    body = _strip_html_fence(final.get("report_html", ""))
    full_html = HTML_SHELL.format(body=body)

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(full_html, encoding="utf-8")

    alloc = final.get("allocation")
    print(f"\nFull report saved to: {report_path}")
    print(f"Open in browser: file://{report_path}")
    if alloc is not None:
        print(f"\nTotal penalty: {alloc.total_penalty}")
        print(f"Summary: {alloc.summary}")
