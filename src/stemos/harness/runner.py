from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stemos.harness.builder import MaterializedHarness
from stemos.harness.memory import MemoryStore
from stemos.harness.role_runner import RoleRunner
from stemos.scenarios.schema import TaskCase


class HarnessRunResult(BaseModel):
    case_id: str
    final_output: str
    outputs: dict[str, str] = Field(default_factory=dict)
    traces: list[dict[str, Any]] = Field(default_factory=list)
    cost_estimate: float = 0.0
    blocked: bool = False
    block_reason: str | None = None


class HarnessRunner:
    def __init__(self, role_runner: RoleRunner | None = None):
        self.role_runner = role_runner or RoleRunner()

    def run_case(self, harness: MaterializedHarness, case: TaskCase) -> HarnessRunResult:
        memory = MemoryStore(harness.memory_layout)
        outputs: dict[str, str] = {}
        traces: list[dict[str, Any]] = []
        attempts = int(harness.retry_policy.get("max_attempts", 1))
        attempts = max(attempts, 1)
        if harness.environment.required_artifacts:
            traces.append(
                {
                    "event": "environment_materialized",
                    "artifacts": list(harness.environment.required_artifacts),
                }
            )

        for step in harness.workflow:
            role = harness.roles[step.role]
            step_input = self.collect_inputs(case, outputs, step.input_from)
            result = self.role_runner.run(role, step, step_input, memory, harness)
            output_key = step.output_key or step.id
            outputs[output_key] = result
            outputs[step.id] = result
            traces.append(
                {
                    "event": "workflow_step",
                    "step_id": step.id,
                    "role": role.name,
                    "output_key": output_key,
                    "attempts": attempts,
                }
            )
            gate_traces, gate_failure = self._run_quality_gates(harness, outputs)
            traces.extend(gate_traces)
            if gate_failure:
                return HarnessRunResult(
                    case_id=case.id,
                    final_output=self.extract_final(outputs),
                    outputs=outputs,
                    traces=traces,
                    cost_estimate=self._estimate_cost(traces),
                    blocked=True,
                    block_reason=gate_failure,
                )

        return HarnessRunResult(
            case_id=case.id,
            final_output=self.extract_final(outputs),
            outputs=outputs,
            traces=traces,
            cost_estimate=self._estimate_cost(traces),
        )

    def collect_inputs(
        self, case: TaskCase, outputs: dict[str, str], input_from: list[str]
    ) -> dict[str, Any]:
        selected = {key: outputs[key] for key in input_from if key in outputs}
        return {"case_input": case.input, "previous_outputs": selected}

    def extract_final(self, outputs: dict[str, str]) -> str:
        if "final_output" in outputs:
            return outputs["final_output"]
        if not outputs:
            return ""
        return next(reversed(outputs.values()))

    def _run_quality_gates(
        self, harness: MaterializedHarness, outputs: dict[str, str]
    ) -> tuple[list[dict[str, Any]], str | None]:
        if "final_output" not in outputs:
            return [], None

        final_output = self.extract_final(outputs)
        final = final_output.lower()
        traces: list[dict[str, Any]] = []
        for gate in harness.quality_gates:
            if not gate.required:
                continue
            passed = True
            missing: list[str] = []
            generated_tool = None

            checker_tools = harness.tool_registry.get(["requirement_sections_checker"])
            if checker_tools:
                generated_tool = checker_tools[0].name
                result = checker_tools[0].run(
                    {
                        "output": final_output,
                        "requirements": harness.scenario.expected_output.requirements,
                    }
                )
                passed = bool(result.get("passed"))
                missing = [str(item) for item in result.get("missing", [])]
            elif gate.check_type == "schema":
                missing = self._missing_required_sections(harness, final)
                passed = not missing

            trace = {
                "event": "quality_gate",
                "gate": gate.name,
                "check_type": gate.check_type,
                "passed": passed,
                "missing": missing,
            }
            if generated_tool:
                trace["generated_tool"] = generated_tool
            traces.append(trace)

            if not passed:
                return traces, f"Required quality gate failed: {gate.name}"
        return traces, None

    def _estimate_cost(self, traces: list[dict[str, Any]]) -> float:
        workflow_cost = sum(1 for trace in traces if trace.get("event") == "workflow_step") * 0.01
        gate_cost = sum(1 for trace in traces if trace.get("event") == "quality_gate") * 0.002
        return round(workflow_cost + gate_cost, 4)

    def _missing_required_sections(
        self, harness: MaterializedHarness, output: str
    ) -> list[str]:
        missing: list[str] = []
        checks = {
            "summary": ["summary"],
            "step": ["steps", "1.", "- "],
            "final": ["final answer", "final output", "recommendation"],
            "acceptance": ["acceptance criteria", "acceptance"],
            "qa": ["qa report", "quality review", "review"],
            "decision": ["decision log", "decision"],
        }
        for requirement in harness.scenario.expected_output.requirements:
            req = requirement.lower()
            passed = True
            for key, needles in checks.items():
                if key in req:
                    passed = any(needle in output for needle in needles)
                    break
            if not passed:
                missing.append(requirement)
        return missing
