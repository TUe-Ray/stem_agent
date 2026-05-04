from __future__ import annotations

from typing import Any

from stemos.genome.models import RoleSpec, WorkflowStep
from stemos.harness.builder import MaterializedHarness
from stemos.harness.memory import MemoryStore
from stemos.nucleus.model_client import ModelClient


class RoleRunner:
    def __init__(self, model_client: ModelClient | None = None):
        self.model_client = model_client or ModelClient(offline=True)

    def run(
        self,
        role: RoleSpec,
        step: WorkflowStep,
        input_payload: dict[str, Any],
        memory: MemoryStore,
        harness: MaterializedHarness,
    ) -> str:
        if self.model_client.offline:
            return self._offline_role_response(role, step, input_payload, harness)

        prompt = self._build_role_prompt(role, step, input_payload, harness)
        tools = harness.tool_registry.get(role.allowed_tools)
        response = self.model_client.call(prompt, tools=tools)
        return str(response)

    def _build_role_prompt(
        self,
        role: RoleSpec,
        step: WorkflowStep,
        input_payload: dict[str, Any],
        harness: MaterializedHarness,
    ) -> str:
        return (
            f"Scenario: {harness.scenario.model_dump(mode='json')}\n"
            f"Role: {role.name}\nInstructions: {role.instructions}\n"
            f"Step: {step.action}\nInput: {input_payload}\n"
        )

    def _offline_role_response(
        self,
        role: RoleSpec,
        step: WorkflowStep,
        input_payload: dict[str, Any],
        harness: MaterializedHarness,
    ) -> str:
        request = self._extract_request(input_payload)
        if "understand" in step.id:
            return f"Intent: {request}"
        if step.id == "review_against_requirements":
            draft = self._previous_output(input_payload, "solve_task", "final_output")
            return self._review_output(draft, harness)
        if step.id == "revise_final_output":
            draft = self._previous_output(input_payload, "solve_task", "final_output")
            review_notes = self._previous_output(input_payload, "review_notes")
            return self._revise_output(draft, review_notes)
        if any(token in role.name.lower() for token in ["checker", "reviewer", "qa"]):
            return "Quality review: check summary, concrete steps, final answer, and scenario requirements."
        return self._draft_output(request, harness)

    def _draft_output(self, request: str, harness: MaterializedHarness) -> str:
        genome = harness.genome
        enhanced = bool(genome.self_evaluation.get("enabled")) or bool(genome.quality_gates)
        environment_aware = bool(genome.environment.required_artifacts)

        if not enhanced and not environment_aware:
            return (
                f"Start by choosing one practical priority for: {request}. "
                "Write down the desired outcome, pick the first action, and schedule a quick check."
            )

        sections = [
            "## Summary",
            f"Create a focused, useful response for: {request}.",
            "",
        ]

        if environment_aware:
            sections.extend(
                [
                    "## Acceptance Criteria",
                    "- The output names the goal.",
                    "- The output gives concrete steps.",
                    "- The output includes a final answer.",
                    "",
                ]
            )

        sections.extend(
            [
                "## Steps",
                "1. Clarify the desired outcome in one sentence.",
                "2. Break the work into the next visible actions.",
                "3. Check the draft against the scenario requirements.",
                "4. Deliver the smallest complete final answer.",
                "",
            ]
        )

        if environment_aware:
            sections.extend(
                [
                    "## QA Report",
                    "Quality review: summary, steps, and final answer are present.",
                    "",
                ]
            )

        if environment_aware:
            sections.extend(
                [
                    "## Decision Log",
                    "- Use a compact structured answer instead of a long exploratory exchange.",
                    "",
                ]
            )

        sections.extend(
            [
                "## Final Answer",
                "Use the plan above as the operating path, then revise only if a required section is missing.",
            ]
        )
        return "\n".join(sections)

    def _review_output(self, draft: str, harness: MaterializedHarness) -> str:
        missing = []
        lower_draft = draft.lower()
        for requirement in harness.scenario.expected_output.requirements:
            if not self._requirement_hint_present(requirement, lower_draft):
                missing.append(requirement)
        if not missing:
            missing_text = "none"
        else:
            missing_text = "; ".join(missing)
        return (
            "## Review Notes\n"
            f"- Checked {len(harness.scenario.expected_output.requirements)} scenario requirements.\n"
            f"- Missing sections: {missing_text}.\n"
            "- Recommendation: revise final output only where required sections are absent."
        )

    def _revise_output(self, draft: str, review_notes: str) -> str:
        if "## Review Notes Applied" in draft:
            return draft
        return "\n\n".join(
            [
                draft.rstrip(),
                "## Review Notes Applied",
                review_notes.strip() or "- No review notes were provided.",
                "Final answer checked against the evolved review loop.",
            ]
        )

    def _extract_request(self, input_payload: dict[str, Any]) -> str:
        case_input = input_payload.get("case_input", {})
        for key in ["user_request", "task", "request", "input"]:
            if key in case_input:
                return str(case_input[key])
        if case_input:
            return " ".join(str(value) for value in case_input.values())
        return str(input_payload)

    def _previous_output(self, input_payload: dict[str, Any], *keys: str) -> str:
        previous_outputs = input_payload.get("previous_outputs", {})
        for key in keys:
            if key in previous_outputs:
                return str(previous_outputs[key])
        if previous_outputs:
            return str(next(reversed(previous_outputs.values())))
        return ""

    def _requirement_hint_present(self, requirement: str, output: str) -> bool:
        requirement = requirement.lower()
        hints = {
            "summary": ["summary"],
            "step": ["steps", "1.", "- "],
            "final": ["final answer", "final output", "recommendation"],
            "acceptance": ["acceptance criteria", "acceptance"],
            "qa": ["qa report", "quality review", "review"],
            "decision": ["decision log", "decision"],
        }
        for key, needles in hints.items():
            if key in requirement:
                return any(needle in output for needle in needles)
        important_words = [word for word in requirement.split() if len(word) >= 5]
        return any(word in output for word in important_words)
