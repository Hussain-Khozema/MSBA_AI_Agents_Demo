from __future__ import annotations
import json
from functools import lru_cache
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from prompts import AUDITOR_PROMPT, CONTEXT_EXTRACTION_PROMPT, REPORT_PROMPT
from schemas import AuditReport, PlaybookExtraction


# ---------------------------------------------------------------------------
# LLMs are constructed lazily on first call so importing this module does
# not require an OPENAI_API_KEY (important for the deterministic test suite,
# which never touches an LLM).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _context_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
        tags=["msba-demo", "multi-corridor", "context-agent"],
        metadata={"repo": "MSBA_AI_Agents_Demo", "stage": "context_extraction"},
    )


@lru_cache(maxsize=1)
def _auditor_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
        tags=["msba-demo", "multi-corridor", "auditor-agent"],
        metadata={"repo": "MSBA_AI_Agents_Demo", "stage": "audit"},
    )


@lru_cache(maxsize=1)
def _report_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        tags=["msba-demo", "multi-corridor", "report-agent"],
        metadata={"repo": "MSBA_AI_Agents_Demo", "stage": "report"},
    )


# ---------------------------------------------------------------------------
# ContextAgent
# ---------------------------------------------------------------------------

def run_context_agent(playbook_markdown: str) -> PlaybookExtraction:
    """Extract the SeeWeeS playbook into a validated PlaybookExtraction.

    Uses OpenAI structured outputs (method="json_schema") so the model
    response is guaranteed to match the pydantic schema. If anything is
    structurally wrong, this raises - we fail fast rather than feeding
    bad data into the deterministic pipeline.
    """
    structured = _context_llm().with_structured_output(
        PlaybookExtraction,
        method="json_schema",
    )
    msg = CONTEXT_EXTRACTION_PROMPT.format_messages(playbook_markdown=playbook_markdown)
    result = structured.invoke(msg)
    if not isinstance(result, PlaybookExtraction):
        result = PlaybookExtraction.model_validate(result)
    return result


# ---------------------------------------------------------------------------
# AuditorAgent
# ---------------------------------------------------------------------------

def run_auditor_agent(findings: list, plan_context: Dict[str, Any]) -> AuditReport:
    """Layer executive narrative on top of deterministic audit findings.

    The deterministic engine produces pass/warn/fail with locked check_id,
    title, status, and severity. The LLM rewrites description + recommendation
    and produces the headline + overall_status fields. Schema-enforced.
    """
    structured = _auditor_llm().with_structured_output(
        AuditReport,
        method="json_schema",
    )
    findings_json = json.dumps(findings, default=str, indent=2)
    plan_json = json.dumps(plan_context, default=str, indent=2)
    msg = AUDITOR_PROMPT.format_messages(
        findings_json=findings_json,
        plan_context_json=plan_json,
    )
    result = structured.invoke(msg)
    if not isinstance(result, AuditReport):
        result = AuditReport.model_validate(result)
    return result


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------

def run_report_agent(payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(payload, default=str, indent=2)
    msg = REPORT_PROMPT.format_messages(payload_json=payload_json)
    return _report_llm().invoke(msg).content
