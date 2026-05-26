from __future__ import annotations

from stem_agent.nucleus.model_client import ModelClient
from stem_agent.nucleus.prompts import NUCLEUS_SYSTEM_PROMPT
from stem_agent.nucleus.schemas import TaskDiagnosis
from stem_agent.scenarios.schema import Scenario


class ScenarioInterpreter:
    """Turns scenario signals into a structured task diagnosis."""

    def __init__(self, model_client: ModelClient | None = None):
        self.model_client = model_client or ModelClient()

    def interpret(self, scenario: Scenario) -> TaskDiagnosis:
        if self.model_client.test_mode:
            return self._test_interpretation(scenario)

        schema = TaskDiagnosis.model_json_schema()
        payload = {
            "scenario": {
                "name": scenario.name,
                "description": scenario.scenario.description,
                "task_class": scenario.scenario.task_class,
                "constraints": scenario.constraints,
                "available_builtin_tools": scenario.available_builtin_tools,
            },
            "instruction": (
                "Read the scenario as environmental signals and produce a "
                "TaskDiagnosis JSON object."
            ),
        }
        scenario.signal_policy.assert_no_layer2_leak(str(payload))
        result = self.model_client.call(
            str(payload),
            response_schema=schema,
            system_prompt=NUCLEUS_SYSTEM_PROMPT,
            temperature=0.8,
            role="nucleus",
        )
        try:
            return TaskDiagnosis.model_validate(result)
        except Exception:
            # Provider returned incomplete JSON — use test interpretation
            # as a safe fallback (e.g. DeepSeek without structured output)
            self.model_client.record_structured_output_repair()
            return self._test_interpretation(scenario)

    def _test_interpretation(self, scenario: Scenario) -> TaskDiagnosis:
        requirements = " ".join(scenario.expected_output.requirements).lower()
        task_type = scenario.scenario.task_class or "general task operation"
        capabilities = ["structured drafting", "requirement checking"]
        failure_modes = ["missing required output sections", "low actionability"]

        if "artifact" in requirements or "acceptance" in requirements:
            capabilities.append("workspace artifact planning")
            failure_modes.append("losing intermediate task decisions")

        return TaskDiagnosis(
            task_type=task_type,
            expected_task_solving_pattern=(
                "Read the input, extract requirements, draft an answer, "
                "check it against scenario requirements, and return a final output."
            ),
            likely_failure_modes=failure_modes,
            likely_needed_capabilities=capabilities,
            initial_architecture_hypothesis=(
                "Start with a Founder role, then evolve self-evaluation, "
                "environment artifacts, and review steps only if fitness shows gaps."
            ),
            initial_evaluation_hypothesis=(
                "Use immutable Guardian fitness for coverage, usefulness, format, "
                "robustness, cost, and complexity; use genome self-evaluation only "
                "inside candidate harnesses."
            ),
        )
