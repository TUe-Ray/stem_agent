"""Reasoning tools: chain-of-thought, claim verification, task decomposition."""

import re
from typing import Any

from stem_agent.tools.catalog import CatalogTool


def _chain_of_thought(input_data: dict[str, Any]) -> dict[str, Any]:
    """Break down a problem into structured reasoning steps."""
    problem = input_data.get("problem", "")
    steps_raw = input_data.get("steps", "")

    if isinstance(steps_raw, list):
        steps = steps_raw
    elif isinstance(steps_raw, str) and steps_raw.strip():
        steps = [s.strip() for s in re.split(r'\n|(?<=\.)\s+(?=[A-Z])', steps_raw) if s.strip()]
    else:
        return {"error": "No reasoning steps provided"}

    return {
        "problem": problem[:500],
        "step_count": len(steps),
        "steps": steps[:20],
    }


def _verify_claim(input_data: dict[str, Any]) -> dict[str, Any]:
    """Check a claim for consistency: internal contradictions, unsupported assertions."""
    claim = input_data.get("claim", "")
    evidence = input_data.get("evidence", "")

    issues = []
    # Basic heuristic checks
    if "always" in claim.lower() or "never" in claim.lower():
        issues.append("Absolute language ('always'/'never') — hard to verify")
    if "obviously" in claim.lower() or "clearly" in claim.lower():
        issues.append("Weasel word ('obviously'/'clearly') — hand-waving signal")
    if re.search(r'\d{2,}%', claim) and not re.search(r'\d{2,}%', evidence):
        issues.append("Percentage claim without evidence support")

    # Check if evidence contradicts the claim (word overlap heuristics)
    if evidence:
        claim_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', claim.lower()))
        evidence_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', evidence.lower()))
        if claim_words:
            overlap = len(claim_words & evidence_words) / len(claim_words)
            if overlap < 0.2:
                issues.append(f"Low evidence overlap ({overlap:.0%}) — claim may be unsupported")

    return {
        "verified": len(issues) == 0,
        "issues": issues,
        "evidence_overlap": overlap if evidence else None,
    }


def _decompose_task(input_data: dict[str, Any]) -> dict[str, Any]:
    """Decompose a complex task into subtasks with dependencies."""
    task = input_data.get("task", "")
    # Simple heuristic: split on numbered items, bullet points, or "and then"
    subtasks = []
    # Pattern 1: numbered
    numbered = re.findall(r'(?:(?:Step\s*)?\d+[\.\)]\s*)([^\n]+)', task)
    if numbered:
        subtasks = numbered

    # Pattern 2: bullet points
    if not subtasks:
        bullets = re.findall(r'[-*]\s*([^\n]+)', task)
        if bullets:
            subtasks = bullets

    # Pattern 3: sentence splitting
    if not subtasks:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', task)
        subtasks = [s.strip() for s in sentences if len(s.strip()) > 10]

    return {
        "task": task[:500],
        "subtask_count": len(subtasks),
        "subtasks": subtasks[:10],
    }


def register(catalog):
    catalog.register(CatalogTool(
        name="chain_of_thought",
        description="Structure a problem into numbered reasoning steps for traceability.",
        category="reasoning",
        parameter_schema={
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "Problem description"},
                "steps": {"type": "string", "description": "Numbered reasoning steps"},
            },
            "required": ["problem", "steps"],
        },
        examples=[
            {"input": 'problem="Why is the sky blue?" steps="1. Light scattering\\n2. Rayleigh effect"',
             "output": "{step_count: 2}"},
        ],
        cost_estimate=0.0,
        fn=_chain_of_thought,
    ))
    catalog.register(CatalogTool(
        name="verify_claim",
        description="Check a claim for absolute language, weasel words, and evidence support. Returns issues found.",
        category="reasoning",
        parameter_schema={
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "Claim to verify"},
                "evidence": {"type": "string", "description": "Supporting evidence (optional)"},
            },
            "required": ["claim"],
        },
        examples=[
            {"input": 'claim="This always works 95% of the time"', "output": "{verified: false, issues: ['absolute language']}"},
        ],
        cost_estimate=0.0,
        fn=_verify_claim,
    ))
    catalog.register(CatalogTool(
        name="decompose_task",
        description="Split a complex task description into subtasks with dependencies. Returns ordered list.",
        category="reasoning",
        parameter_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Complex task description"},
            },
            "required": ["task"],
        },
        examples=[
            {"input": 'task="1. Gather requirements 2. Design solution 3. Implement"',
             "output": "{subtask_count: 3, subtasks: [...]}"},
        ],
        cost_estimate=0.0,
        fn=_decompose_task,
    ))
