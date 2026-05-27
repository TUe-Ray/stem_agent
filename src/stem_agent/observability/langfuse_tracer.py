"""
Langfuse observability integration for stem_agent.

Follows Langfuse best practices (see langfuse/skills references/instrumentation.md):
  1. Prefer framework integrations → use OpenAI observer for auto-capture
  2. @observe() decorator for nested spans
  3. session_id groups trace by run, user_id by scenario
  4. Tags: model, role, generation
  5. flush() before exit

Usage:
    1. Sign up at https://cloud.langfuse.com (free Hobby tier)
    2. Create project → API keys → set env vars:
       export LANGFUSE_PUBLIC_KEY="pk-lf-..."
       export LANGFUSE_SECRET_KEY="sk-lf-..."
       export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
    3. stem_agent picks them up automatically

When env vars are not set: zero overhead (tracer is a no-op).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any


def _get_openai_class():
    """Return Langfuse-observed OpenAI class, or plain OpenAI as fallback.
    
    Called lazily when env vars are already loaded.
    """
    try:
        from langfuse.openai import OpenAI as LfOpenAI
        return LfOpenAI
    except Exception:
        from openai import OpenAI
        return OpenAI


# ── Global state ──

_langfuse: Any = None  # lazy-init
_langfuse_init_done = False


def _ensure_langfuse():
    """Lazy-init Langfuse client. Returns True if configured."""
    global _langfuse, _langfuse_init_done
    if _langfuse_init_done:
        return _langfuse is not None
    _langfuse_init_done = True
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        return False
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")),
        )
        return True
    except Exception:
        return False


def is_enabled() -> bool:
    return _ensure_langfuse()


# Lazy OpenAI class — use get_openai() not LangfuseOpenAI directly
LangfuseOpenAI = None  # Set lazily; use get_openai() instead


def get_openai():
    """Return the appropriate OpenAI class (Langfuse-wrapped or plain)."""
    if _ensure_langfuse():
        return _get_openai_class()
    from openai import OpenAI
    return OpenAI


# ── Run-level trace ──

def start_run(run_id: str, scenario: str, model: str) -> None:
    """Start a top-level trace for the evolution run."""
    if not _ensure_langfuse():
        return
    try:
        trace = _langfuse.trace(
            name=f"evolve:{run_id}",
            session_id=run_id,
            user_id=scenario,
            tags=[model, scenario, run_id],
            metadata={"run_id": run_id, "scenario": scenario, "model": model},
        )
        _current_trace_id = trace.id
        _current_run_id = run_id
    except Exception:
        pass


# ── Nested spans (via @observe decorator) ──

def observe(*, name: str = "", as_type: str = "span", **kwargs):
    """Decorator that wraps a function in a Langfuse span/generation.
    
    Falls back to passthrough when Langfuse is not configured.
    """
    if not _ensure_langfuse():
        def passthrough(fn):
            return fn
        return passthrough
    from langfuse.decorators import observe as _observe
    return _observe(name=name, as_type=as_type, **kwargs)


# ── Tool call events ──

def log_tool_call(
    tool_name: str,
    args: dict[str, Any],
    result_summary: str,
    success: bool,
    duration_ms: float,
) -> None:
    """Log a tool execution as an event on the current span."""
    if not _ensure_langfuse():
        return
    try:
        span = _langfuse.get_current_span()
        if span:
            span.event(
                name=f"tool:{tool_name}",
                metadata={
                    "tool": tool_name,
                    "args": str(args)[:200],
                    "result": result_summary[:200],
                    "success": success,
                    "duration_ms": round(duration_ms, 1),
                },
            )
    except Exception:
        pass


# ── Scores ──

def log_score(name: str, value: float, comment: str = "") -> None:
    """Log an evaluation score on the current trace."""
    if not _ensure_langfuse():
        return
    try:
        trace = _langfuse.get_current_trace()
        if not trace:
            return
        _langfuse.score(
            trace_id=_current_trace_id,
            name=name,
            value=value,
            comment=comment,
        )
    except Exception:
        pass

def log_case_score(case_id: str, resolved: bool, patch_applied: bool, score: float) -> None:
    """Log per-case SWE-bench score."""
    log_score(
        name=f"case:{case_id}",
        value=score,
        comment=f"resolved={resolved} patch_applied={patch_applied}",
    )


# ── Flush ──

def flush() -> None:
    """Flush pending events. Call at end of run."""
    if _ensure_langfuse():
        try:
            _langfuse.flush()
        except Exception:
            pass
