from __future__ import annotations

from stem_agent.genome.models import Genome
from stem_agent.kernel.signal_policy import NucleusSignal, SignalPolicy
from stem_agent.nucleus.operators import CrossoverMutation, FirstOrderMutation, MutationOperator
from stem_agent.nucleus.model_client import ModelClient
from stem_agent.nucleus.failure_analyzer import FailurePattern
from stem_agent.nucleus.nucleus import build_nucleus_prompt
from stem_agent.nucleus.prompts import NUCLEUS_SYSTEM_PROMPT
from stem_agent.nucleus.schemas import MutationPlan, MutationProposal
from stem_agent.scenarios.schema import Scenario


class MutationPlanner:
    def __init__(self, model_client: ModelClient | None = None):
        self.model_client = model_client or ModelClient()

    def propose_mutations(
        self,
        *,
        scenario: Scenario,
        genome: Genome,
        generation: int,
        last_score: float,
        best_score: float,
        failure_patterns: list[str],
        structured_failure_patterns: list[FailurePattern] | None = None,
        budget_remaining: float,
        lineage_summary: str,
        operator: MutationOperator | None = None,
        archive: object | None = None,
        mutation_history: list[NucleusSignal] | None = None,
        signal_policy: SignalPolicy | None = None,
    ) -> MutationPlan:
        operator = operator or FirstOrderMutation()
        signal_policy = signal_policy or scenario.signal_policy
        mutation_history = mutation_history or []
        operator_plan = self._operator_plan(
            operator=operator,
            archive=archive,
            scenario=scenario,
            genome=genome,
            failure_patterns=failure_patterns,
        )
        if operator_plan is not None:
            self._audit_test_plan(
                scenario,
                genome,
                operator,
                operator_plan,
                mutation_history,
                signal_policy,
            )
            return operator_plan

        if self.model_client.test_mode:
            plan = self._test_plan(
                scenario=scenario,
                genome=genome,
                failure_patterns=failure_patterns,
                structured_failure_patterns=structured_failure_patterns,
                lineage_summary=lineage_summary,
            )
            self._audit_test_plan(
                scenario,
                genome,
                operator,
                plan,
                mutation_history,
                signal_policy,
            )
            return plan

        _ = (last_score, best_score, budget_remaining, lineage_summary)
        prompt = build_nucleus_prompt(
            scenario_description=self._scenario_description(scenario),
            current_genome=genome.model_dump(mode="json"),
            mutation_history=mutation_history,
            signal_policy=signal_policy,
            failure_patterns=structured_failure_patterns,
        )
        result = self.model_client.call(
            prompt,
            response_schema=MutationPlan.model_json_schema(),
            system_prompt=NUCLEUS_SYSTEM_PROMPT,
            temperature=0.8,
            role="nucleus",
        )
        if isinstance(result, dict) and "mutation_plan" in result:
            self.model_client.record_structured_output_repair()
            result = result["mutation_plan"]
        # If the LLM returned a single mutation proposal at the root, wrap it.
        if isinstance(result, dict) and "mutation_type" in result and "proposed_mutations" not in result:
            self.model_client.record_structured_output_repair()
            result = {"summary": result.get("rationale", ""), "proposed_mutations": [result]}
        try:
            return MutationPlan.model_validate(result)
        except Exception as exc:
            self.model_client.record_structured_output_repair()
            raise RuntimeError("Nucleus returned an invalid mutation plan.") from exc

    def _operator_plan(
        self,
        *,
        operator: MutationOperator,
        archive: object | None,
        scenario: Scenario,
        genome: Genome,
        failure_patterns: list[str],
    ) -> MutationPlan | None:
        if isinstance(operator, FirstOrderMutation):
            return None
        genome_data = genome.model_dump(mode="json")
        history = [{"scenario": scenario.model_dump(mode="json"), "failure_patterns": failure_patterns}]
        if isinstance(operator, CrossoverMutation) and archive is not None and len(archive) >= 2:
            _, best_genome, _ = archive.best()
            _, random_genome = archive.sample_parent(strategy="random")
            replacement = operator.propose(best_genome, random_genome, self.model_client)
        else:
            replacement = operator.propose(genome_data, history, self.model_client)
        replacement["genome_version"] = int(genome_data.get("genome_version", 0)) + 1
        return MutationPlan(
            summary=f"{operator.name} operator proposed a whole-genome candidate.",
            failure_patterns=failure_patterns,
            proposed_mutations=[
                MutationProposal(
                    mutation_type="replace_genome",
                    target="genome",
                    rationale=f"{operator.name} selected because search stagnated or archive diversity was useful.",
                    expected_improvement="Explore a non-greedy candidate outside the current local hill climb.",
                    risk="Whole-genome replacement may remove useful organs and will be safety-checked before evaluation.",
                    patch={"genome": replacement},
                )
            ],
        )

    def _audit_test_plan(
        self,
        scenario: Scenario,
        genome: Genome,
        operator: MutationOperator,
        plan: MutationPlan,
        mutation_history: list[NucleusSignal],
        signal_policy: SignalPolicy,
    ) -> None:
        _ = operator
        prompt = build_nucleus_prompt(
            scenario_description=self._scenario_description(scenario),
            current_genome=genome.model_dump(mode="json"),
            mutation_history=mutation_history,
            signal_policy=signal_policy,
            failure_patterns=plan.failure_patterns,
        )
        self.model_client.audit_call(
            role="nucleus",
            system_prompt=NUCLEUS_SYSTEM_PROMPT,
            prompt=prompt,
            response=plan.model_dump(mode="json"),
        )

    def _scenario_description(self, scenario: Scenario) -> str:
        return f"{scenario.scenario.task_class}: {scenario.scenario.description}"

    def _test_plan(
        self,
        *,
        scenario: Scenario,
        genome: Genome,
        failure_patterns: list[str],
        structured_failure_patterns: list[FailurePattern] | None,
        lineage_summary: str,
    ) -> MutationPlan:
        mutations: list[MutationProposal] = []
        targeted_mutations: list[MutationProposal] = []
        primary = structured_failure_patterns[0] if structured_failure_patterns else None
        categories = {pattern.category for pattern in structured_failure_patterns or []}
        if primary and primary.category == "reasoning_or_calculation_error":
            if not self._workflow_has(genome, "verification_step"):
                targeted_mutations.append(MutationProposal(
                    mutation_type="add_workflow_step", target="workflow",
                    rationale="Primary failures indicate reasoning or calculation mistakes.",
                    expected_improvement="Add a verification pass before final answer.",
                    risk="Adds one workflow step and extra cost.",
                    patch={"id":"verification_step","role":"Founder","action":"Verify calculations and critical claims before final output.","input_from":["solve_task"],"output_key":"verification_notes"},
                ))
        elif primary and primary.category == "cost_pressure":
            targeted_mutations.append(MutationProposal(
                mutation_type="modify_retry_policy", target="retry_policy",
                rationale="Primary failures indicate budget pressure.",
                expected_improvement="Reduce retry budget and improve cost efficiency.",
                risk="May reduce recovery from transient errors.",
                patch={"max_retries":1},
            ))
        elif primary and primary.category == "quality_gate_block":
            targeted_mutations.append(MutationProposal(
                mutation_type="edit_quality_gate", target="quality_gates[0]",
                rationale="Primary failures indicate quality gate block behavior.",
                expected_improvement="Tune quality gate to avoid blocking valid responses.",
                risk="Could reduce strictness if tuned poorly.",
                patch={"required": True},
            ))

        if not genome.self_evaluation.get("enabled"):
            mutations.append(
                MutationProposal(
                    mutation_type="modify_self_evaluation",
                    target="self_evaluation",
                    rationale="Metric feedback shows the harness lacks an explicit self-review loop.",
                    expected_improvement="Create evaluator-visible review intent before adding more complex organs.",
                    risk="The harness may spend extra effort reviewing its own draft.",
                    patch={
                        "enabled": True,
                        "rubric": scenario.expected_output.requirements,
                    },
                )
            )
        elif (
            not genome.environment.required_artifacts
            and "artifact_gap" in categories
            and self._scenario_wants_artifacts(scenario)
        ):
            mutations.append(
                MutationProposal(
                    mutation_type="modify_environment",
                    target="environment",
                    rationale="The harness needs persistent task artifacts to preserve acceptance criteria, draft, QA, decisions, and final output.",
                    expected_improvement="Improve task-operation structure and validation score.",
                    risk="More artifacts add complexity, so Guardian fitness should penalize overgrowth.",
                    patch={
                        "workspace_layout": ["artifacts"],
                        "required_artifacts": [
                            "artifacts/acceptance_criteria.md",
                            "artifacts/draft_output.md",
                            "artifacts/qa_report.md",
                            "artifacts/decision_log.md",
                            "artifacts/final_output.md",
                        ],
                        "artifact_purpose": {
                            "artifacts/acceptance_criteria.md": "Track scenario requirements.",
                            "artifacts/draft_output.md": "Hold the candidate draft.",
                            "artifacts/qa_report.md": "Record mutable self-evaluation notes.",
                            "artifacts/decision_log.md": "Explain important harness decisions.",
                            "artifacts/final_output.md": "Store the final response.",
                        },
                        "file_templates": {
                            "artifacts/acceptance_criteria.md": "# Acceptance Criteria\n",
                            "artifacts/draft_output.md": "# Draft Output\n",
                            "artifacts/qa_report.md": "# QA Report\n",
                            "artifacts/decision_log.md": "# Decision Log\n",
                            "artifacts/final_output.md": "# Final Output\n",
                        },
                        "cleanup_policy": "keep_run_artifacts",
                    },
                )
            )
        elif not self._workflow_has(genome, "review_against_requirements"):
            mutations.append(
                MutationProposal(
                    mutation_type="add_workflow_step",
                    target="workflow",
                    rationale="The harness currently produces final output without explicit review.",
                    expected_improvement="Improve requirement coverage and reduce missing sections.",
                    risk="Adds cost and complexity.",
                    patch={
                        "id": "review_against_requirements",
                        "role": "Founder",
                        "action": "Review the draft or final output against the scenario requirements and write missing sections or revision notes.",
                        "input_from": ["solve_task"],
                        "output_key": "review_notes",
                    },
                )
            )
        elif not self._workflow_has(genome, "revise_final_output"):
            mutations.append(
                MutationProposal(
                    mutation_type="add_workflow_step",
                    target="workflow",
                    rationale="Review notes should affect the delivered output.",
                    expected_improvement="Create a real revise-before-final loop.",
                    risk="May duplicate content if poorly executed.",
                    patch={
                        "id": "revise_final_output",
                        "role": "Founder",
                        "action": "Revise the final answer using review notes, scenario constraints, and success criteria.",
                        "input_from": ["solve_task", "review_notes"],
                        "output_key": "final_output",
                    },
                )
            )
        elif not genome.quality_gates:
            mutations.append(
                MutationProposal(
                    mutation_type="add_quality_gate",
                    target="quality_gates",
                    rationale="The harness should not finish without checking required output structure.",
                    expected_improvement="Reduce missing required sections.",
                    risk="May reject valid outputs if too strict.",
                    patch={
                        "name": "required_sections_gate",
                        "description": "Check that the output contains required scenario sections before final delivery.",
                        "check_type": "schema",
                        "required": True,
                    },
                )
            )
        elif "requirement_sections_checker" not in self._generated_tool_names(genome):
            mutations.append(
                MutationProposal(
                    mutation_type="create_tool",
                    target="tools.generated",
                    rationale="Add a safe checker tool that can support evolved quality gates.",
                    expected_improvement="Improve self-review evidence while staying inside the sandbox.",
                    risk="Adds one generated tool and a small complexity penalty.",
                    patch={
                        "name": "requirement_sections_checker",
                        "description": "Check whether output appears to cover scenario requirements.",
                        "code": self._safe_requirement_checker_code(),
                        "test_code": self._safe_requirement_checker_test(),
                    },
                )
            )
        elif (
            "RequirementChecker" not in {role.name for role in genome.roles}
            and "RequirementChecker" not in lineage_summary
        ):
            mutations.append(
                MutationProposal(
                    mutation_type="add_role",
                    target="roles",
                    rationale="Add an explicit reviewer role after self-evaluation has stabilized.",
                    expected_improvement="May improve review quality.",
                    risk="Could add complexity without improving immutable fitness.",
                    patch={
                        "name": "RequirementChecker",
                        "description": "Checks whether output satisfies scenario requirements.",
                        "instructions": "Compare the draft against scenario requirements and report missing sections.",
                        "allowed_tools": ["call_model"],
                    },
                )
            )


        if not mutations and targeted_mutations:
            mutations.extend(targeted_mutations)
        return MutationPlan(
            summary=(
                "Test Nucleus plan based on repeated failure patterns."
                if mutations
                else "Current genome is good enough; freeze."
            ),
            failure_patterns=failure_patterns,
            proposed_mutations=mutations[: scenario.evolution.max_mutations_per_generation],
        )

    def _workflow_has(self, genome: Genome, step_id: str) -> bool:
        return any(step.id == step_id for step in genome.workflow)

    def _scenario_wants_artifacts(self, scenario: Scenario) -> bool:
        text = " ".join(scenario.expected_output.requirements).lower()
        return any(
            token in text
            for token in [
                "acceptance",
                "qa",
                "quality review",
                "decision",
                "artifact",
                "verification",
                "risk",
                "fallback",
            ]
        )

    def _generated_tool_names(self, genome: Genome) -> set[str]:
        names: set[str] = set()
        for spec in genome.tools.get("generated", []) or []:
            if isinstance(spec, dict) and spec.get("name"):
                names.add(str(spec["name"]))
            elif getattr(spec, "name", None):
                names.add(str(spec.name))
        return names

    def _safe_requirement_checker_code(self) -> str:
        return (
            "from typing import Any, Dict\n\n"
            "def run(input_data: Dict[str, Any]) -> Dict[str, Any]:\n"
            "    output = str(input_data.get(\"output\", \"\")).lower()\n"
            "    requirements = input_data.get(\"requirements\", [])\n"
            "    missing = []\n"
            "    for req in requirements:\n"
            "        req_text = str(req).lower()\n"
            "        important_words = [\n"
            "            word.strip(\".,:;!?()[]{}\")\n"
            "            for word in req_text.split()\n"
            "            if len(word.strip(\".,:;!?()[]{}\")) >= 5\n"
            "        ]\n"
            "        if important_words and not any(word in output for word in important_words):\n"
            "            missing.append(req)\n"
            "    total = max(len(requirements), 1)\n"
            "    score = 1.0 - (len(missing) / total)\n"
            "    return {\"passed\": len(missing) == 0, \"missing\": missing, \"score\": score}\n"
        )

    def _safe_requirement_checker_test(self) -> str:
        return (
            "from requirement_sections_checker import run\n\n"
            "def test_requirement_sections_checker_detects_missing_requirement():\n"
            "    result = run({\n"
            "        \"output\": \"Summary: hello\",\n"
            "        \"requirements\": [\"Must include a short summary\", \"Must include concrete steps\"],\n"
            "    })\n"
            "    assert result[\"passed\"] is False\n"
            "    assert \"Must include concrete steps\" in result[\"missing\"]\n\n"
            "def test_requirement_sections_checker_accepts_covered_output():\n"
            "    result = run({\n"
            "        \"output\": \"Summary with concrete steps and final answer\",\n"
            "        \"requirements\": [\"Must include summary\", \"Must include concrete steps\"],\n"
            "    })\n"
            "    assert result[\"score\"] >= 0.5\n"
        )
