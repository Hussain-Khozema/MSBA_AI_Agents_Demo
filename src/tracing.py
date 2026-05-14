from __future__ import annotations
import os

def init_langsmith_tracing() -> None:
    """
    Enables LangSmith tracing via env vars.
    Kept as a small explicit init so it's obvious in workshops.
    """
    # If user didn't set it, default to off (safer)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    # Optional: provide a default project name if not set
    os.environ.setdefault("LANGCHAIN_PROJECT", "MSBA_AI_Agents_Demo")
