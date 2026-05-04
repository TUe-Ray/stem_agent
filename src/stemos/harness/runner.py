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

        for step in harness.workflow:
            role = harness.roles[step.role]
            step_input = self.collect_inputs(case, outputs, step.input_from)
            result = self.role_runner.run(role, step, step_input, memory, harness)
            output_key = step.output_key or step.id
            outputs[output_key] = result
            traces.append(
                {
                    "step_id": step.id,
                    "role": role.name,
                    "output_key": output_key,
                    "attempts": attempts,
                }
            )
            gate_result = self._run_quality_gates(harness, outputs)
            if gate_result:
                return HarnessRunResult(
                    case_id=case.id,
                    final_output=result,
                    outputs=outputs,
                    traces=traces,
                    cost_estimate=self._estimate_cost(traces),
                    blocked=True,
                    block_reason=gate_result,
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
    ) -> str | None:
        final = self.extract_final(outputs).lower()
        for gate in harness.quality_gates:
            if not gate.required:
                continue
            if gate.check_type == "schema" and "final" not in final:
                return f"Required schema quality gate failed: {gate.name}"
        return None

    def _estimate_cost(self, traces: list[dict[str, Any]]) -> float:
        return round(len(traces) * 0.01, 4)
