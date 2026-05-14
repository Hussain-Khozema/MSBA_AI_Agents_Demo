"""Orchestrates ContextAgent: read the playbook markdown, run the LLM
extractor with strict pydantic validation, and persist the result to
data/augmented/extracted_context.json as an audit trail.

The extracted JSON is the single source of truth for everything the
playbook contains. Downstream code reads from this file (via
playbook_loader), never directly from the markdown again.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from agents import run_context_agent
from schemas import PlaybookExtraction


EXTRACTED_FILENAME = "extracted_context.json"


def extract_and_persist(
    playbook_md_path: str | Path,
    augmented_dir: str | Path,
    force: bool = False,
) -> PlaybookExtraction:
    """Run ContextAgent over the playbook and write the extracted JSON.

    Args:
        playbook_md_path: path to the playbook .md file
        augmented_dir: directory to write extracted_context.json into
        force: if True, re-run the agent even when a cached extraction exists.
            If False (default) and the file exists, we load it from disk.
            This makes development cheap (no LLM call on every run) while
            still letting graders trigger a fresh extraction.
    """
    augmented_dir = Path(augmented_dir)
    augmented_dir.mkdir(parents=True, exist_ok=True)
    out_path = augmented_dir / EXTRACTED_FILENAME

    if out_path.exists() and not force:
        with out_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return PlaybookExtraction.model_validate(cached)

    markdown = Path(playbook_md_path).read_text(encoding="utf-8")
    extraction: PlaybookExtraction = run_context_agent(markdown)

    payload: Dict[str, Any] = extraction.model_dump()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return extraction
