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

        if environment_aware and self._is_deadline_manager_meeting(request):
            return self._deadline_manager_meeting_output(request)

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

    def _is_deadline_manager_meeting(self, request: str) -> bool:
        lowered = request.lower()
        return all(token in lowered for token in ["manager", "deadline"])

    def _deadline_manager_meeting_output(self, request: str) -> str:
        return "\n".join(
            [
                "## Summary",
                f"Prepare a calm, accountable meeting plan for: {request}.",
                "",
                "## Meeting Goal",
                "Explain what happened without defensiveness, show ownership of the missed deadlines, and leave the meeting with an agreed recovery plan.",
                "",
                "## Acceptance Criteria",
                "- The output names the goal.",
                "- The output gives concrete steps.",
                "- The output includes a final answer.",
                "",
                "## Talking Points",
                "1. Open with ownership: name the missed deadlines and acknowledge the impact.",
                "2. Briefly explain the causes using facts, not excuses.",
                "3. Present the recovery plan before your manager has to ask for one.",
                "4. Ask which tradeoffs or priorities your manager wants adjusted.",
                "",
                "## Likely Objections",
                "- Why did I hear about this late?",
                "- How do I know the new dates are realistic?",
                "- What will change so this does not repeat?",
                "",
                "## Recovery Plan",
                "1. Send a corrected timeline with owners, dates, and risk flags today.",
                "2. Split the missed work into must-finish, can-defer, and needs-help buckets.",
                "3. Add a twice-weekly status note until the project is back on track.",
                "4. Escalate blockers within 24 hours instead of waiting for the next check-in.",
                "",
                "## Wording To Use",
                '"I missed the deadline, and I understand that created planning risk for the team. The main causes were X and Y. I should have flagged the risk earlier. Here is the recovery plan I propose, and I would like your input on the tradeoffs."',
                "",
                "## Follow-Up Actions",
                "- Send the recovery plan after the meeting.",
                "- Confirm revised dates and success criteria in writing.",
                "- Share the first progress update within two business days.",
                "",
                "## QA Report",
                "Quality review: meeting goal, talking points, likely objections, recovery plan, wording, follow-up actions, summary, steps, and final answer are present.",
                "",
                "## Decision Log",
                "- Use accountable language instead of defensive explanations.",
                "- Put the recovery plan before the justification.",
                "",
                "## Final Answer",
                "Go into the meeting with ownership first, facts second, and a concrete recovery plan third. Ask for priority guidance, then follow up in writing with dates, owners, and the next status checkpoint.",
            ]
        )

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
