from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal


class MutationOperator(ABC):
    name: str
    order: Literal["zero", "first", "hyper"]

    @abstractmethod
    def propose(self, genome: dict, history: list[dict], llm: Any) -> dict:
        """Return a mutated genome."""


class ZeroOrderMutation(MutationOperator):
    name = "zero_order"
    order: Literal["zero"] = "zero"

    def propose(self, genome: dict, history: list[dict], llm: Any) -> dict:
        scenario = next((item.get("scenario", {}) for item in reversed(history) if item.get("scenario")), {})
        scenario_name = scenario.get("scenario", {}).get("name") or genome.get("scenario_name") or "scenario"
        fresh = deepcopy(genome)
        fresh["name"] = f"{scenario_name}_zero_order_seed"
        fresh["roles"] = [
            {
                "name": "solver",
                "description": "Fresh task solver created without relying on the prior genome shape.",
                "instructions": "Solve the problem from the scenario description. Show useful intermediate work and give the final answer.",
                "allowed_tools": ["call_model"],
            }
        ]
        fresh["workflow"] = [
            {
                "id": "solve_from_scratch",
                "role": "solver",
                "action": "Solve the task from scratch using the scenario input and fixed evaluation criteria.",
                "input_from": [],
                "output_key": "final_output",
            }
        ]
        fresh["quality_gates"] = []
        fresh["tools"] = {"builtin": ["call_model"], "generated": []}
        fresh["self_evaluation"] = {"enabled": True, "rubric": _scenario_rubric(scenario)}
        return fresh


class FirstOrderMutation(MutationOperator):
    name = "first_order"
    order: Literal["first"] = "first"

    def propose(self, genome: dict, history: list[dict], llm: Any) -> dict:
        return deepcopy(genome)


class HyperMutation(MutationOperator):
    name = "hyper"
    order: Literal["hyper"] = "hyper"

    def propose(self, genome: dict, history: list[dict], llm: Any) -> dict:
        mutated = deepcopy(genome)
        roles = list(mutated.get("roles") or [])
        if not any(role.get("name") == "StrategyAuditor" for role in roles if isinstance(role, dict)):
            roles.append(
                {
                    "name": "StrategyAuditor",
                    "description": "Looks for a new class of mutation when ordinary edits stall.",
                    "instructions": "Identify the weakest case pattern and force the final response to address it explicitly.",
                    "allowed_tools": ["call_model"],
                }
            )
        mutated["roles"] = roles
        workflow = list(mutated.get("workflow") or [])
        if not any(step.get("id") == "audit_strategy_shift" for step in workflow if isinstance(step, dict)):
            workflow.append(
                {
                    "id": "audit_strategy_shift",
                    "role": "StrategyAuditor",
                    "action": "Audit the current solving strategy and add one concrete correction before final output.",
                    "input_from": [workflow[-1]["id"]] if workflow else [],
                    "output_key": "strategy_audit",
                }
            )
        mutated["workflow"] = workflow
        return mutated


class CrossoverMutation(MutationOperator):
    name = "crossover"
    order: Literal["first"] = "first"

    def propose(self, genome_a: dict, genome_b: dict | list[dict], llm: Any) -> dict:
        if not isinstance(genome_b, dict):
            return deepcopy(genome_a)
        child = deepcopy(genome_a)
        child["tools"] = deepcopy(genome_b.get("tools", genome_a.get("tools", {})))
        child["workflow"] = deepcopy(genome_a.get("workflow", []))
        child["roles"] = deepcopy(genome_a.get("roles", []))
        return child


def _scenario_rubric(scenario: dict) -> list[str]:
    criteria = scenario.get("evaluation_criteria") or []
    if criteria:
        return [str(item.get("name", item)) if isinstance(item, dict) else str(item) for item in criteria]
    expected = scenario.get("expected_output") or {}
    return [str(item) for item in expected.get("requirements", [])]
