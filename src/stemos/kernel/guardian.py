from __future__ import annotations

from copy import deepcopy
import re
from pathlib import Path
from typing import Any

from stemos.genome.models import EnvironmentSpec, Genome, QualityGate, RoleSpec, WorkflowStep
from stemos.kernel.evaluator import EvaluationResult, GuardianFitnessEvaluator
from stemos.kernel.sandbox import Sandbox
from stemos.kernel.validators import ValidationResult
from stemos.nucleus.schemas import MutationProposal
from stemos.scenarios.schema import Scenario


PROTECTED_TARGET_MARKERS = {
    "src/stemos/kernel",
    "kernel/",
    "guardian.py",
    "evaluator.py",
    "sandbox.py",
    "budget.py",
    "versioning.py",
    "validators.py",
    "hidden evaluation",
    "hidden_evaluation",
    "lineage",
    "rollback",
    "budget limits",
    "budget",
    "mutation validation",
}


class Guardian:
    """Immutable safety kernel for mutation validation and promotion decisions."""

    def __init__(self, evaluator: GuardianFitnessEvaluator | None = None):
        self.evaluator = evaluator or GuardianFitnessEvaluator()
        self.sandbox = Sandbox()

    def validate_mutation(
        self,
        mutation: MutationProposal,
        genome: Genome | None = None,
        scenario: Scenario | None = None,
    ) -> ValidationResult:
        marker_text = f"{mutation.target} {mutation.rationale} {mutation.patch}".lower()
        for marker in PROTECTED_TARGET_MARKERS:
            if marker in marker_text:
                return ValidationResult.reject(f"Mutation touches protected kernel boundary: {marker}")

        if mutation.mutation_type in {"create_tool", "edit_tool"}:
            validation = self._validate_tool_mutation(mutation)
            if not validation.allowed:
                return validation

        if "command" in mutation.patch:
            validation = self.sandbox.validate_shell_command(mutation.patch)
            if not validation.allowed:
                return validation

        if genome and scenario:
            if mutation.mutation_type == "add_workflow_step":
                if len(genome.workflow) + 1 > scenario.evolution.max_workflow_steps:
                    return ValidationResult.reject("Workflow exceeds configured max_workflow_steps")
            if mutation.mutation_type == "add_role":
                if len(genome.roles) + 1 > scenario.evolution.max_roles:
                    return ValidationResult.reject("Roles exceed configured max_roles")
            if mutation.mutation_type == "modify_environment":
                new_artifacts = mutation.patch.get("required_artifacts", [])
                total_artifacts = len(set(genome.environment.required_artifacts + list(new_artifacts)))
                if total_artifacts > scenario.evolution.max_environment_artifacts:
                    return ValidationResult.reject(
                        "Environment exceeds configured max_environment_artifacts"
                    )

        return ValidationResult.allow("mutation is inside mutable genome boundary")

    def activate_generated_tool(
        self, mutation: MutationProposal, tool_dir: Path
    ) -> tuple[ValidationResult, MutationProposal]:
        """Write, statically check, test, and path-fill a generated tool mutation."""
        if mutation.mutation_type not in {"create_tool", "edit_tool"}:
            return ValidationResult.allow("not a generated tool mutation"), mutation

        validation = self.validate_mutation(mutation)
        if not validation.allowed:
            return validation, mutation

        patch = deepcopy(mutation.patch)
        name = patch["name"]
        code = patch.get("code")
        test_code = patch.get("test_code")
        if not code or not test_code:
            return ValidationResult.reject("Generated tool activation requires code and test_code"), mutation

        tool_dir.mkdir(parents=True, exist_ok=True)
        (tool_dir / "__init__.py").write_text("", encoding="utf-8")
        tool_path = tool_dir / f"{name}.py"
        test_path = tool_dir / f"test_{name}.py"
        tool_path.write_text(code, encoding="utf-8")
        test_path.write_text(test_code, encoding="utf-8")

        static_validation = self.sandbox.validate_generated_tool_code(code)
        if not static_validation.allowed:
            return static_validation, mutation

        test_validation = self.sandbox.run_generated_tool_tests(tool_dir)
        if not test_validation.allowed:
            return test_validation, mutation

        patch["path"] = str(tool_path)
        patch["test_path"] = str(test_path)
        activated = mutation.model_copy(update={"patch": patch}, deep=True)
        return ValidationResult.allow("generated tool activated after static checks and tests"), activated

    def apply_mutation_safely(self, genome: Genome, mutation: MutationProposal) -> Genome:
        mutated = genome.model_copy(deep=True)
        mutated.genome_version += 1
        patch = deepcopy(mutation.patch)

        if mutation.mutation_type == "add_role":
            mutated.roles.append(RoleSpec.model_validate(patch))
        elif mutation.mutation_type == "edit_role":
            self._edit_named(mutated.roles, mutation.target, patch)
        elif mutation.mutation_type == "add_workflow_step":
            mutated.workflow.append(WorkflowStep.model_validate(patch))
        elif mutation.mutation_type == "edit_workflow_step":
            self._edit_named(mutated.workflow, mutation.target, patch, id_field="id")
        elif mutation.mutation_type == "remove_workflow_step":
            mutated.workflow = [step for step in mutated.workflow if step.id != mutation.target]
        elif mutation.mutation_type == "add_quality_gate":
            mutated.quality_gates.append(QualityGate.model_validate(patch))
        elif mutation.mutation_type == "edit_quality_gate":
            self._edit_named(mutated.quality_gates, mutation.target, patch)
        elif mutation.mutation_type == "modify_memory_schema":
            mutated.memory.setdefault("schema", {}).update(patch)
        elif mutation.mutation_type == "modify_retry_policy":
            mutated.retry_policy.update(patch)
        elif mutation.mutation_type == "modify_self_evaluation":
            mutated.self_evaluation.update(patch)
        elif mutation.mutation_type == "modify_stop_rule":
            mutated.stop_rule.update(patch)
        elif mutation.mutation_type == "modify_environment":
            mutated.environment = self._merge_environment(mutated.environment, patch)
        elif mutation.mutation_type in {"create_tool", "edit_tool"}:
            generated = list(mutated.tools.get("generated", []) or [])
            generated.append(
                {
                    "name": patch["name"],
                    "description": patch.get("description", ""),
                    "kind": "generated",
                    "path": patch.get("path"),
                    "test_path": patch.get("test_path"),
                }
            )
            mutated.tools["generated"] = generated
        return Genome.model_validate(mutated.model_dump(mode="json"))

    def evaluate_candidate(self, *args: Any, **kwargs: Any) -> EvaluationResult:
        return self.evaluator.evaluate(*args, **kwargs)

    def should_promote(self, old_score: float, new_score: float, min_delta: float) -> bool:
        return (new_score - old_score) >= min_delta

    def rollback(self, version_id: str) -> None:
        # File rollback is handled by VersionStore; this method documents the kernel boundary.
        _ = version_id

    def should_freeze(
        self,
        history: list[EvaluationResult],
        patience: int = 2,
        min_delta: float = 0.03,
    ) -> bool:
        if len(history) <= patience:
            return False
        best_before_window = max(result.promotion_score for result in history[:-patience])
        best_recent = max(result.promotion_score for result in history[-patience:])
        return best_recent - best_before_window < min_delta

    def _validate_tool_mutation(self, mutation: MutationProposal) -> ValidationResult:
        patch = mutation.patch
        name = str(patch.get("name", ""))
        if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            return ValidationResult.reject("Generated tool name must be a valid Python identifier")
        if not patch.get("test_code") and not patch.get("test_path"):
            return ValidationResult.reject("Generated tools require tests before activation")
        if patch.get("code"):
            validation = self.sandbox.validate_generated_tool_code(patch["code"])
            if not validation.allowed:
                return validation
        return ValidationResult.allow("generated tool mutation includes tests and static checks")

    def _edit_named(
        self,
        items: list[Any],
        target: str,
        patch: dict[str, Any],
        *,
        id_field: str = "name",
    ) -> None:
        for index, item in enumerate(items):
            if getattr(item, id_field) == target:
                data = item.model_dump(mode="json")
                data.update(patch)
                items[index] = item.__class__.model_validate(data)
                return
        raise ValueError(f"Cannot edit missing target: {target}")

    def _merge_environment(
        self, environment: EnvironmentSpec, patch: dict[str, Any]
    ) -> EnvironmentSpec:
        data = environment.model_dump(mode="json")
        for key in ["workspace_layout", "required_artifacts"]:
            if key in patch:
                data[key] = list(dict.fromkeys(data.get(key, []) + list(patch[key])))
        for key in ["artifact_purpose", "file_templates"]:
            if key in patch:
                data.setdefault(key, {}).update(patch[key])
        if "cleanup_policy" in patch:
            data["cleanup_policy"] = patch["cleanup_policy"]
        return EnvironmentSpec.model_validate(data)
