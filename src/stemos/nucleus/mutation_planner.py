from __future__ import annotations

from stemos.genome.models import Genome
from stemos.nucleus.model_client import ModelClient
from stemos.nucleus.prompts import NUCLEUS_SYSTEM_PROMPT
from stemos.nucleus.schemas import MutationPlan, MutationProposal
from stemos.scenarios.schema import Scenario


class MutationPlanner:
    def __init__(self, model_client: ModelClient | None = None):
        self.model_client = model_client or ModelClient(offline=True)

    def propose_mutations(
        self,
        *,
        scenario: Scenario,
        genome: Genome,
        generation: int,
        last_score: float,
        best_score: float,
        failure_patterns: list[str],
        budget_remaining: float,
        lineage_summary: str,
    ) -> MutationPlan:
        if self.model_client.offline:
            return self._deterministic_plan(
                scenario=scenario,
                genome=genome,
                failure_patterns=failure_patterns,
                lineage_summary=lineage_summary,
            )

        payload = {
            "scenario": scenario.model_dump(mode="json"),
            "current_genome": genome.model_dump(mode="json"),
            "generation": generation,
            "last_score": last_score,
            "best_score": best_score,
            "failure_patterns": failure_patterns,
            "budget_remaining": budget_remaining,
            "lineage_summary": lineage_summary,
        }
        result = self.model_client.call(
            NUCLEUS_SYSTEM_PROMPT + "\n" + str(payload),
            response_schema=MutationPlan.model_json_schema(),
        )
        if isinstance(result, dict) and "mutation_plan" in result:
            self.model_client.record_structured_output_repair()
            result = result["mutation_plan"]
        try:
            return MutationPlan.model_validate(result)
        except Exception:
            self.model_client.record_structured_output_repair()
            return self._deterministic_plan(
                scenario=scenario,
                genome=genome,
                failure_patterns=failure_patterns,
                lineage_summary=lineage_summary,
            )

    def _deterministic_plan(
        self,
        *,
        scenario: Scenario,
        genome: Genome,
        failure_patterns: list[str],
        lineage_summary: str,
    ) -> MutationPlan:
        mutations: list[MutationProposal] = []

        if not genome.self_evaluation.get("enabled"):
            mutations.append(
                MutationProposal(
                    mutation_type="modify_self_evaluation",
                    target="self_evaluation",
                    rationale="Previous runs missed required output sections.",
                    expected_improvement="Improve requirement coverage without changing Guardian fitness.",
                    risk="The harness may spend extra effort reviewing its own draft.",
                    patch={
                        "enabled": True,
                        "rubric": scenario.expected_output.requirements,
                    },
                )
            )
            mutations.append(
                MutationProposal(
                    mutation_type="create_tool",
                    target="tools.generated",
                    rationale="Try to inspect workspace files for extra context.",
                    expected_improvement="Could improve artifact awareness.",
                    risk="Unsafe filesystem access must be blocked by Guardian.",
                    patch={
                        "name": "unsafe_workspace_probe",
                        "description": "Unsafe demo tool that Guardian should reject.",
                        "code": "import os\n\ndef run(input_data):\n    return {'cwd': os.getcwd()}\n",
                        "test_code": "def test_unsafe_workspace_probe():\n    assert True\n",
                    },
                )
            )
        elif not genome.environment.required_artifacts:
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

        return MutationPlan(
            summary=(
                "Offline deterministic Nucleus plan based on repeated failure patterns."
                if mutations
                else "Current genome is good enough; freeze."
            ),
            failure_patterns=failure_patterns,
            proposed_mutations=mutations[: scenario.evolution.max_mutations_per_generation],
        )

    def _workflow_has(self, genome: Genome, step_id: str) -> bool:
        return any(step.id == step_id for step in genome.workflow)

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
