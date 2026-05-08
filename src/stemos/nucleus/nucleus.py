from __future__ import annotations

from dataclasses import asdict
import json
import random
from typing import Any

from stemos.kernel.signal_policy import NucleusSignal, SignalPolicy
from stemos.nucleus.operators import (
    CrossoverMutation,
    FirstOrderMutation,
    HyperMutation,
    MutationOperator,
    ZeroOrderMutation,
)


def select_operator(archive, stagnation_count: int) -> MutationOperator:
    if stagnation_count >= 5:
        return HyperMutation()
    if stagnation_count >= 3:
        return ZeroOrderMutation()
    if stagnation_count >= 1 and len(archive) >= 3 and random.random() < 0.25:
        return CrossoverMutation()
    return FirstOrderMutation()


def build_nucleus_prompt(
    scenario_description: str,
    current_genome: dict,
    mutation_history: list[NucleusSignal],
    signal_policy: SignalPolicy,
    reusable_skills: list[Any] | None = None,
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

Reusable Skills Retrieved From Skill Library:
{_format_reusable_skills(reusable_skills or [])}

Your job: propose ONE mutation to the genome that may improve harness performance.
You do NOT know the specific validation cases. You do NOT know per-case scores.
Base your proposal on the genome structure and the directional feedback above.

Output JSON only:
{{
    "mutation_type": "<one of: modify_role | add_workflow_step | modify_self_evaluation | add_quality_gate | create_tool | modify_retry_policy | zero_order_redesign | crossover>",
    "target_field": "<genome field path, e.g. roles[0].instruction>",
    "rationale": "<why this change, based on mutation history>",
    "new_value": <the proposed new value>
}}
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


def _format_reusable_skills(skills: list[Any]) -> str:
    if not skills:
        return "No reusable skills retrieved."
    lines: list[str] = []
    for skill in skills[:5]:
        if hasattr(skill, "model_dump"):
            data = skill.model_dump(mode="json")
        elif isinstance(skill, dict):
            data = skill
        else:
            continue
        tags = ", ".join(str(tag) for tag in data.get("domain_tags", [])[:5])
        lines.append(
            "- "
            f"{data.get('name')} | {data.get('skill_type')} | "
            f"lift={float(data.get('score_lift', 0.0)):.3f} | "
            f"origin={data.get('origin_scenario_name')} | "
            f"tags={tags or 'none'} | "
            f"{data.get('description', '')}"
        )
    return "\n".join(lines) if lines else "No reusable skills retrieved."


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
