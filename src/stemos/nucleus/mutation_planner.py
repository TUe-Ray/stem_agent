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
        return MutationPlan.model_validate(result)

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
        elif (
            not genome.environment.required_artifacts
            and self._scenario_benefits_from_environment(scenario)
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

    def _scenario_benefits_from_environment(self, scenario: Scenario) -> bool:
        text = " ".join(
            [
                scenario.name,
                scenario.scenario.description,
                *scenario.expected_output.requirements,
                *scenario.constraints,
            ]
        ).lower()
        return any(
            token in text
            for token in ["artifact", "acceptance", "qa", "decision", "operator", "task"]
        )
