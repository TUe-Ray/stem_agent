from __future__ import annotations

from dataclasses import asdict
import json
import random
from typing import Any

from stem_agent.kernel.signal_policy import NucleusSignal, SignalPolicy
from stem_agent.nucleus.operators import (
    CrossoverMutation,
    FirstOrderMutation,
    HyperMutation,
    MutationOperator,
    ZeroOrderMutation,
)


def select_operator(archive, stagnation_count: int) -> MutationOperator:
    if stagnation_count >= 8:
        return HyperMutation()
    if stagnation_count >= 5:
        return ZeroOrderMutation()
    if stagnation_count >= 2 and len(archive) >= 3 and random.random() < 0.20:
        return CrossoverMutation()
    return FirstOrderMutation()


def build_nucleus_prompt(
    scenario_description: str,
    current_genome: dict,
    mutation_history: list[NucleusSignal],
    signal_policy: SignalPolicy,
    failure_patterns: list[Any] | None = None,
    max_mutations: int = 1,
) -> str:
    history_lines = []
    for sig in mutation_history[-5:]:
        if (
            signal_policy.layer_1_enabled
            and sig.aggregate_score is not None
            and sig.previous_aggregate_score is not None
        ):
            line = (
                f"Generation {sig.generation}: {sig.mutation_type} -> "
                f"{sig.direction} (score: {sig.previous_aggregate_score:.3f} -> "
                f"{sig.aggregate_score:.3f})"
            )
        else:
            line = f"Generation {sig.generation}: {sig.mutation_type} -> {sig.direction}"
        history_lines.append(line)

    prompt = f"""You are Nucleus, a genome mutation proposer for a stem agent framework.

TASK CLASS:
{scenario_description}

CURRENT GENOME STRUCTURE:
{format_genome_for_nucleus(current_genome)}

RECENT MUTATION HISTORY:
{chr(10).join(history_lines) if history_lines else "No history yet."}

STRUCTURED FAILURE PATTERNS:
{_format_failure_patterns(failure_patterns or [])}

Your job: propose 1 to {max(1, max_mutations)} mutations to the genome that may improve harness performance.
You do NOT know the specific validation cases. You do NOT know per-case scores.
Base your proposal on the genome structure and the directional feedback above.
Prefer metric-deficit fixes over retry-policy changes. Only propose retry changes for
explicit cost, budget, blocked-run, or failure-recovery patterns. For structured answer
tasks, prefer this order when applicable: self-evaluation, review step, revise-final
step, lightweight quality gate, then a safe generated checker tool. Avoid adding roles
or whole-genome redesigns when complexity pressure is present.

Output JSON only as a MutationPlan:
{{
  "summary": "<short plan summary>",
  "failure_patterns": ["<visible aggregate failure pattern>"],
  "proposed_mutations": [
    {{
      "mutation_type": "<one of: add_role | edit_role | add_workflow_step | edit_workflow_step | remove_workflow_step | create_tool | edit_tool | add_quality_gate | edit_quality_gate | modify_memory_schema | modify_retry_policy | modify_self_evaluation | modify_stop_rule | modify_environment>",
      "target": "<genome field path, e.g. workflow or self_evaluation>",
      "rationale": "<why this change, based on mutation history>",
      "expected_improvement": "<which visible aggregate signal should improve>",
      "risk": "<short risk>",
      "patch": {{ "<valid patch for the mutation type>": "..." }}
    }}
  ]
}}

Patch requirements:
- Mutations must not create duplicate workflow step ids, role names, quality gate names, or generated tool names.
- add_quality_gate patch: {{"name": "...", "description": "...", "check_type": "schema", "required": true}}
- add_workflow_step patch: {{"id": "...", "role": "<existing role>", "action": "...", "input_from": ["..."], "output_key": "..."}}
- modify_self_evaluation patch: {{"enabled": true, "rubric": ["..."]}}
"""
    signal_policy.assert_no_layer2_leak(prompt)
    return prompt


def format_genome_for_nucleus(current_genome: dict[str, Any]) -> str:
    structure = {
        "name": current_genome.get("name"),
        "genome_version": current_genome.get("genome_version"),
        "scenario_name": current_genome.get("scenario_name"),
        "roles": [
            {
                "name": role.get("name"),
                "description": role.get("description"),
                "allowed_tools": role.get("allowed_tools", []),
            }
            for role in current_genome.get("roles", [])
            if isinstance(role, dict)
        ],
        "workflow": [
            {
                "id": step.get("id"),
                "role": step.get("role"),
                "input_from": step.get("input_from", []),
                "output_key": step.get("output_key"),
            }
            for step in current_genome.get("workflow", [])
            if isinstance(step, dict)
        ],
        "quality_gate_names": [
            gate.get("name")
            for gate in current_genome.get("quality_gates", [])
            if isinstance(gate, dict)
        ],
        "tool_names": _tool_names(current_genome.get("tools", {})),
        "retry_policy": current_genome.get("retry_policy", {}),
        "environment_artifact_count": len(
            (current_genome.get("environment") or {}).get("required_artifacts", [])
        ),
    }
    return json.dumps(structure, sort_keys=True, indent=2)


def nucleus_signal_to_dict(signal: NucleusSignal) -> dict[str, Any]:
    return asdict(signal)


def _tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, dict):
        return []
    names: list[str] = []
    for item in tools.get("builtin", []) or []:
        names.append(str(item))
    for item in tools.get("generated", []) or []:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def _format_failure_patterns(failure_patterns: list[Any]) -> str:
    if not failure_patterns:
        return "No structured failure patterns provided."
    lines: list[str] = []
    for pattern in failure_patterns[:5]:
        if hasattr(pattern, "model_dump"):
            data = pattern.model_dump(mode="json")
        elif isinstance(pattern, dict):
            data = pattern
        else:
            continue
        lines.append(
            "- "
            f"category={data.get('category', 'unknown')} | "
            f"severity={float(data.get('severity', 0.0)):.4f} | "
            f"count={int(data.get('count', 0))} | "
            f"suggested_operator={data.get('suggested_operator') or 'none'} | "
            f"kind={str(data.get('kind', ''))}"
        )
    return "\\n".join(lines) if lines else "No structured failure patterns provided."
