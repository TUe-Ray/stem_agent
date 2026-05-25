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

        if sig.weakest_metrics:
            line += f"\n    weakest metrics: {', '.join(sig.weakest_metrics)}"
        if sig.metric_breakdown:
            mini = {k: v for k, v in sig.metric_breakdown.items() if v < 0.5}
            if mini:
                line += f"\n    low metrics: {mini}"
        if sig.last_rejection_summary:
            line += f"\n    last rejection: {sig.last_rejection_summary[:200]}"
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

Important: create_tool mutations are a powerful way to add custom validation logic
that earns generated_tool_usage and quality_gate_usage scores.
Here is a WORKING EXAMPLE of a create_tool patch that passes all sandbox checks:

  mutation_type: create_tool
  target: tools.generated
  patch:
    name: requirement_sections_checker
    description: Check whether output covers scenario requirements.
    code: |
      from typing import Any, Dict
      def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
          output = str(input_data.get("output", "")).lower()
          requirements = input_data.get("requirements", [])
          missing = []
          for req in requirements:
              req_text = str(req).lower()
              important_words = [
                  word.strip(".,:;!?()[]{{}}")
                  for word in req_text.split()
                  if len(word.strip(".,:;!?()[]{{}}")) >= 5
              ]
              if important_words and not any(word in output for word in important_words):
                  missing.append(req)
          total = max(len(requirements), 1)
          score = 1.0 - (len(missing) / total)
          return {{"passed": len(missing) == 0, "missing": missing, "score": score}}
    test_code: |
      from requirement_sections_checker import run
      def test_detects_missing():
          result = run({{"output": "Summary: hello", "requirements": ["Must include a short summary", "Must include concrete steps"]}})
          assert not result["passed"] and "Must include concrete steps" in result["missing"]
      def test_accepts_covered():
          result = run({{"output": "Summary with concrete steps and final answer", "requirements": ["Must include summary", "Must include concrete steps"]}})
          assert result["score"] >= 0.5

IMPORTANT RULES for generated tool code:
- Functions must be named `run(input_data) -> dict` and return a dict.
- Name must be a valid Python identifier matching `^[A-Za-z_][A-Za-z0-9_]*$`.
- Must NOT import: os, subprocess, socket, requests, httpx, shutil (FORBIDDEN).
- May import from `typing`, `pathlib` (read-only), `json`, `re`, `math`.
- test_code must `from <tool_name> import run` (the tool file is <tool_name>.py).
- code AND test_code are BOTH required — no test_code means automatic rejection.
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
