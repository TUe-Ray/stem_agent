from __future__ import annotations

from copy import deepcopy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from stem_agent.genome.models import EnvironmentSpec, Genome, QualityGate, RoleSpec, WorkflowStep
from stem_agent.harness.builder import MaterializedHarness
from stem_agent.harness.runner import HarnessRunner
from stem_agent.kernel.evaluator import EvaluationResult, GuardianFitnessEvaluator
from stem_agent.kernel.sandbox import Sandbox
from stem_agent.kernel.signal_policy import NucleusSignal, SignalPolicy
from stem_agent.kernel.validators import MutationSafetyValidator, ValidationResult
from stem_agent.nucleus.schemas import MutationProposal
from stem_agent.scenarios.schema import TaskCase
from stem_agent.scenarios.schema import Scenario


PROTECTED_TARGET_MARKERS = {
    "src/stem_agent/kernel",
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

    def __init__(
        self,
        signal_policy: SignalPolicy | None = None,
        evaluator: GuardianFitnessEvaluator | None = None,
        harness_runner: HarnessRunner | None = None,
    ):
        self.evaluator = evaluator or GuardianFitnessEvaluator()
        self.harness_runner = harness_runner or HarnessRunner()
        self._signal_policy = signal_policy or SignalPolicy()
        self._score_history: list[float] = []
        self.sandbox = Sandbox()
        self.run_dir: Path | None = None
        self.hidden_cases_path: Path | None = None
        self.hidden_baseline_score: float | None = None
        self._validation_cases: list[TaskCase] = []

    def configure_run(self, run_dir: str | Path | None) -> None:
        self.run_dir = Path(run_dir) if run_dir else None
        if hasattr(self.evaluator, "configure_run"):
            self.evaluator.configure_run(run_dir)

    def configure_signal_policy(self, signal_policy: SignalPolicy) -> None:
        self._signal_policy = signal_policy

    def produce_nucleus_signal(
        self,
        harness: MaterializedHarness,
        generation: int,
        mutation_type: str,
    ) -> tuple[bool, NucleusSignal]:
        """
        Evaluates internally and returns only the policy-filtered NucleusSignal.
        This compatibility boundary is intentionally aggregate-only.
        """
        eval_result = self._run_evaluation(harness, self._validation_cases)
        current_score = float(eval_result["aggregate_score"])
        hidden_score = self._run_hidden_evaluation(harness, f"generation_{generation:03d}", generation=generation)
        _ = hidden_score
        previous_score = self._score_history[-1] if self._score_history else 0.0
        promoted = self._promotion_decision(current_score, previous_score, eval_result)
        self._score_history.append(current_score)
        signal = self._signal_policy.build_nucleus_signal(
            generation=generation,
            mutation_type=mutation_type,
            current_score=current_score,
            previous_score=previous_score,
        )
        return promoted, signal

    def configure_hidden_evaluation(self, scenario_path: str | Path) -> None:
        root = Path(scenario_path)
        if root.is_file():
            root = root.parent
        self.hidden_cases_path = root / "hidden_cases.jsonl"

    def configure_validation_cases(self, validation_cases: list[TaskCase]) -> None:
        self._validation_cases = list(validation_cases)

    def _run_evaluation(self, harness: MaterializedHarness, cases: list[TaskCase]) -> dict[str, Any]:
        runs = [self.harness_runner.run_case(harness, case) for case in cases]
        result = self.evaluator.evaluate(
            harness.genome,
            harness.scenario,
            [],
            runs,
            run_dir=self.run_dir,
        )
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aggregate_score": result.promotion_score,
            "validation_score": result.validation_score,
            "case_count": len(cases),
        }
        if self.run_dir is not None:
            with (self.run_dir / "eval_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        return item

    def _promotion_decision(
        self,
        current_score: float,
        previous_score: float,
        eval_result: dict[str, Any],
    ) -> bool:
        _ = eval_result
        return self.should_promote(previous_score, current_score, 0.01)

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

        schema_validation = self._validate_mutation_patch_schema(mutation, genome=genome)
        if not schema_validation.allowed:
            return schema_validation

        if genome is not None:
            structural_validation = self._validate_structural_invariants(mutation, genome)
            if not structural_validation.allowed:
                return structural_validation

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

        if genome is not None:
            resulting_validation = self._validate_resulting_genome(mutation, genome)
            if not resulting_validation.allowed:
                return resulting_validation

        return ValidationResult.allow("mutation is inside mutable genome boundary")

    def _validate_mutation_patch_schema(
        self,
        mutation: MutationProposal,
        *,
        genome: Genome | None = None,
    ) -> ValidationResult:
        add_models = {
            "add_role": RoleSpec,
            "add_workflow_step": WorkflowStep,
            "add_quality_gate": QualityGate,
        }
        model = add_models.get(mutation.mutation_type)
        if model is not None:
            try:
                model.model_validate(mutation.patch)
            except ValidationError as exc:
                return ValidationResult.reject(
                    self._mutation_schema_error(mutation.mutation_type, exc)
                )

        if mutation.mutation_type == "replace_genome":
            if "genome" not in mutation.patch:
                return ValidationResult.reject("Invalid replace_genome patch: missing genome")
            try:
                Genome.model_validate(mutation.patch["genome"])
            except ValidationError as exc:
                return ValidationResult.reject(
                    self._mutation_schema_error(mutation.mutation_type, exc)
                )

        if mutation.mutation_type in {"edit_role", "edit_workflow_step", "edit_quality_gate"}:
            if not mutation.target:
                return ValidationResult.reject(
                    f"Invalid {mutation.mutation_type} patch: missing target"
                )
            if genome is not None:
                validation = self._validate_edit_patch_schema(mutation, genome)
                if not validation.allowed:
                    return validation

        if mutation.mutation_type == "remove_workflow_step":
            if not mutation.target:
                return ValidationResult.reject("Invalid remove_workflow_step patch: missing target")
            if genome is not None and not any(step.id == mutation.target for step in genome.workflow):
                return ValidationResult.reject(
                    f"Invalid remove_workflow_step patch: missing workflow step {mutation.target}"
                )

        if mutation.mutation_type == "modify_environment" and genome is not None:
            try:
                self._merge_environment(genome.environment, mutation.patch)
            except (TypeError, ValueError, ValidationError) as exc:
                return ValidationResult.reject(
                    f"Invalid modify_environment patch: {self._exception_summary(exc)}"
                )

        return ValidationResult.allow("mutation patch schema is valid")

    def _validate_structural_invariants(
        self,
        mutation: MutationProposal,
        genome: Genome,
    ) -> ValidationResult:
        patch = mutation.patch
        if mutation.mutation_type == "add_workflow_step":
            step_id = str(patch.get("id", ""))
            if self._workflow_step_exists(genome, step_id):
                return ValidationResult.reject(
                    f"Invalid add_workflow_step patch: duplicate workflow step id {step_id}"
                )
        if mutation.mutation_type == "edit_workflow_step" and "id" in patch:
            step_id = str(patch.get("id", ""))
            if step_id != mutation.target and self._workflow_step_exists(genome, step_id):
                return ValidationResult.reject(
                    f"Invalid edit_workflow_step patch: duplicate workflow step id {step_id}"
                )
        if mutation.mutation_type == "add_role":
            role_name = str(patch.get("name", ""))
            if self._role_exists(genome, role_name):
                return ValidationResult.reject(
                    f"Invalid add_role patch: duplicate role name {role_name}"
                )
        if mutation.mutation_type == "edit_role" and "name" in patch:
            role_name = str(patch.get("name", ""))
            if role_name != mutation.target and self._role_exists(genome, role_name):
                return ValidationResult.reject(
                    f"Invalid edit_role patch: duplicate role name {role_name}"
                )
        if mutation.mutation_type == "add_quality_gate":
            gate_name = str(patch.get("name", ""))
            if self._quality_gate_exists(genome, gate_name):
                return ValidationResult.reject(
                    f"Invalid add_quality_gate patch: duplicate quality gate name {gate_name}"
                )
        if mutation.mutation_type == "edit_quality_gate" and "name" in patch:
            gate_name = str(patch.get("name", ""))
            if gate_name != mutation.target and self._quality_gate_exists(genome, gate_name):
                return ValidationResult.reject(
                    f"Invalid edit_quality_gate patch: duplicate quality gate name {gate_name}"
                )
        if mutation.mutation_type in {"create_tool", "edit_tool"}:
            tool_name = str(patch.get("name", ""))
            if tool_name and tool_name in self._generated_tool_names(genome):
                return ValidationResult.reject(
                    f"Invalid {mutation.mutation_type} patch: duplicate generated tool name {tool_name}"
                )
        return ValidationResult.allow("mutation structural invariants are valid")

    def _validate_resulting_genome(
        self,
        mutation: MutationProposal,
        genome: Genome,
    ) -> ValidationResult:
        try:
            self.apply_mutation_safely(genome, mutation)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            return ValidationResult.reject(
                f"Invalid {mutation.mutation_type} resulting genome: {self._exception_summary(exc)}"
            )
        return ValidationResult.allow("resulting genome structure is valid")

    def validate_candidate_safety(
        self,
        old_genome: Genome,
        new_genome: Genome,
        scenario: Scenario,
        *,
        train_cases: list[dict[str, Any]] | None = None,
        run_dir: str | Path | None = None,
    ) -> ValidationResult:
        target_run_dir = Path(run_dir) if run_dir else self.run_dir
        validator = MutationSafetyValidator(target_run_dir)
        old_data = old_genome.model_dump(mode="json")
        new_data = new_genome.model_dump(mode="json")
        scenario_data = scenario.model_dump(mode="json")
        if train_cases is not None:
            scenario_data["train_cases"] = train_cases

        rubric_validation = validator.validate_self_eval_rubric_change(
            old_data,
            new_data,
            scenario_data,
        )
        if not rubric_validation.allowed:
            return rubric_validation

        stop_rule_validation = validator.validate_stop_rule_change(old_data, new_data)
        if not stop_rule_validation.allowed:
            return stop_rule_validation
        return ValidationResult.allow("candidate passed mutation safety validation")

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
        elif mutation.mutation_type == "replace_genome":
            replacement = Genome.model_validate(patch["genome"])
            replacement.genome_version = mutated.genome_version
            # Preserve previously-promoted quality gates and generated tools
            preserved_tools = dict(mutated.tools)
            preserved_gates = list(mutated.quality_gates)
            mutated = replacement
            # Merge back preserved gates that don't already exist in replacement
            existing_gate_names = {g.name for g in mutated.quality_gates}
            for gate in preserved_gates:
                if gate.name not in existing_gate_names:
                    mutated.quality_gates.append(gate)
            # Merge back preserved generated tools
            existing_tool_names = {t.get("name") for t in mutated.tools.get("generated", []) if isinstance(t, dict)}
            for tool in preserved_tools.get("generated", []) or []:
                name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
                if name and name not in existing_tool_names:
                    mutated.tools.setdefault("generated", []).append(tool)
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

    def _run_hidden_evaluation(
        self,
        harness: MaterializedHarness,
        genome_id: str,
        *,
        generation: int | None = None,
    ) -> float:
        """
        Runs hidden_cases.jsonl through harness.
        Results written to runs/<run_id>/hidden_eval_log.jsonl.
        Never returned to Nucleus. Never used in fitness score.
        """
        if self.run_dir is None or self.hidden_cases_path is None or not self.hidden_cases_path.exists():
            return 0.0
        cases = self._read_hidden_cases(self.hidden_cases_path)
        if not cases:
            return 0.0
        runs = [self.harness_runner.run_case(harness, case) for case in cases]
        result = self.evaluator.evaluate(
            harness.genome,
            harness.scenario,
            runs,
            [],
            run_dir=self.run_dir,
            audit=False,
        )
        score = float(result.train_score)
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
            "genome_id": genome_id,
            "hidden_score": score,
            "case_scores": [
                {"case_id": case.case_id, "score": case.score, "metrics": case.metrics}
                for case in result.case_results
            ],
        }
        with (self.run_dir / "hidden_eval_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        return score

    def should_promote(self, old_score: float, new_score: float, min_delta: float) -> bool:
        return (new_score - old_score) + 1e-9 >= min_delta

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

    def _validate_edit_patch_schema(
        self,
        mutation: MutationProposal,
        genome: Genome,
    ) -> ValidationResult:
        collections: dict[str, tuple[list[Any], str]] = {
            "edit_role": (list(genome.roles), "name"),
            "edit_workflow_step": (list(genome.workflow), "id"),
            "edit_quality_gate": (list(genome.quality_gates), "name"),
        }
        items, id_field = collections[mutation.mutation_type]
        for item in items:
            if getattr(item, id_field) == mutation.target:
                data = item.model_dump(mode="json")
                data.update(mutation.patch)
                try:
                    item.__class__.model_validate(data)
                except ValidationError as exc:
                    return ValidationResult.reject(
                        self._mutation_schema_error(mutation.mutation_type, exc)
                    )
                return ValidationResult.allow("edit mutation patch schema is valid")
        return ValidationResult.reject(
            f"Invalid {mutation.mutation_type} patch: missing target {mutation.target}"
        )

    def _mutation_schema_error(
        self,
        mutation_type: str,
        exc: ValidationError,
    ) -> str:
        details = []
        for error in exc.errors()[:3]:
            loc = ".".join(str(part) for part in error.get("loc", ())) or "patch"
            details.append(f"{loc}: {error.get('msg', 'invalid value')}")
        return f"Invalid {mutation_type} patch: {'; '.join(details)}"

    def _exception_summary(self, exc: Exception) -> str:
        if isinstance(exc, ValidationError):
            details = []
            for error in exc.errors()[:3]:
                loc = ".".join(str(part) for part in error.get("loc", ())) or "genome"
                details.append(f"{loc}: {error.get('msg', 'invalid value')}")
            return "; ".join(details)
        return str(exc) or exc.__class__.__name__

    def _workflow_step_exists(self, genome: Genome, step_id: str) -> bool:
        return any(step.id == step_id for step in genome.workflow)

    def _role_exists(self, genome: Genome, role_name: str) -> bool:
        return any(role.name == role_name for role in genome.roles)

    def _quality_gate_exists(self, genome: Genome, gate_name: str) -> bool:
        return any(gate.name == gate_name for gate in genome.quality_gates)

    def _generated_tool_names(self, genome: Genome) -> set[str]:
        names: set[str] = set()
        for item in genome.tools.get("generated", []) or []:
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
            elif getattr(item, "name", None):
                names.add(str(item.name))
        return names

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

    def _read_hidden_cases(self, path: Path) -> list[TaskCase]:
        cases: list[TaskCase] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(TaskCase.model_validate(json.loads(line)))
        return cases
