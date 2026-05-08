from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from stemos.genome.models import RoleSpec, WorkflowStep
from stemos.harness.builder import MaterializedHarness
from stemos.harness.intra_adapter import IntraTestAdapter
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
    expected_output: str | None = None
    case_input: dict[str, Any] = Field(default_factory=dict)


class HarnessRunner:
    def __init__(self, role_runner: RoleRunner | None = None):
        self.role_runner = role_runner or RoleRunner()

    def run_case(
        self,
        harness: MaterializedHarness,
        case: TaskCase,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
        run_dir: str | Path | None = None,
    ) -> HarnessRunResult:
        memory = MemoryStore(harness.memory_layout)
        outputs: dict[str, str] = {}
        traces: list[dict[str, Any]] = []
        attempts = int(harness.retry_policy.get("max_attempts", 1))
        attempts = max(attempts, 1)
        intra = IntraTestAdapter(run_dir) if run_dir is not None else None
        if harness.environment.required_artifacts:
            self._record_trace(
                traces,
                {
                    "event": "environment_materialized",
                    "artifacts": list(harness.environment.required_artifacts),
                },
                trace_sink,
            )

        for step in harness.workflow:
            role = harness.roles[step.role]
            step_input = self.collect_inputs(case, outputs, step.input_from)
            result = self.role_runner.run(role, step, step_input, memory, harness)
            output_key = step.output_key or step.id
            outputs[output_key] = result
            outputs[step.id] = result
            self._record_trace(
                traces,
                self._workflow_step_trace(
                    role=role,
                    step=step,
                    step_input=step_input,
                    output_key=output_key,
                    output=result,
                    attempts=attempts,
                ),
                trace_sink,
            )
            gate_traces, gate_failure = self._run_quality_gates(harness, outputs)
            for trace in gate_traces:
                self._record_trace(traces, trace, trace_sink)
            if (
                gate_failure
                and intra is not None
                and intra.active(harness.genome)
                and gate_traces
            ):
                gate_failure = self._retry_with_reflection(
                    harness=harness,
                    case=case,
                    memory=memory,
                    outputs=outputs,
                    traces=traces,
                    trace_sink=trace_sink,
                    intra=intra,
                    step=step,
                    role=role,
                    step_input=step_input,
                    output_key=output_key,
                    original_output=result,
                    first_gate_trace=gate_traces[-1],
                )
            if gate_failure:
                if intra is not None:
                    intra.flush()
                return HarnessRunResult(
                    case_id=case.id,
                    final_output=self.extract_final(outputs),
                    outputs=outputs,
                    traces=traces,
                    cost_estimate=self._estimate_cost(traces),
                    blocked=True,
                    block_reason=gate_failure,
                    expected_output=case.expected_output,
                    case_input=dict(case.input),
                )

        if intra is not None:
            intra.flush()
        return HarnessRunResult(
            case_id=case.id,
            final_output=self.extract_final(outputs),
            outputs=outputs,
            traces=traces,
            cost_estimate=self._estimate_cost(traces),
            expected_output=case.expected_output,
            case_input=dict(case.input),
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

    def _record_trace(
        self,
        traces: list[dict[str, Any]],
        trace: dict[str, Any],
        trace_sink: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        traces.append(trace)
        if trace_sink:
            trace_sink(trace)

    def _workflow_step_trace(
        self,
        *,
        role: RoleSpec,
        step: WorkflowStep,
        step_input: dict[str, Any],
        output_key: str,
        output: str,
        attempts: int,
    ) -> dict[str, Any]:
        previous_outputs = step_input.get("previous_outputs", {})
        case_input = step_input.get("case_input", {})
        return {
            "event": "workflow_step",
            "step_id": step.id,
            "role": role.name,
            "role_description": role.description,
            "step_action": step.action,
            "input_from": list(step.input_from),
            "input_keys": {
                "case_input": sorted(str(key) for key in case_input.keys()),
                "previous_outputs": sorted(str(key) for key in previous_outputs.keys()),
            },
            "agent_observation": {
                "goal": step.action,
                "available_context": self._summarize_input(step_input),
                "output_summary": self._summarize_text(output),
                "reasoning_boundary": (
                    "Visible execution state only; hidden chain-of-thought is not recorded."
                ),
            },
            "output_key": output_key,
            "output_summary": self._summarize_text(output),
            "attempts": attempts,
            "status": "completed",
        }

    def _summarize_input(self, step_input: dict[str, Any]) -> str:
        case_input = step_input.get("case_input", {})
        previous_outputs = step_input.get("previous_outputs", {})
        parts: list[str] = []
        if case_input:
            parts.append("case_input=" + self._summarize_mapping(case_input))
        if previous_outputs:
            parts.append("previous_outputs=" + self._summarize_mapping(previous_outputs))
        return "; ".join(parts) if parts else "no prior context"

    def _summarize_mapping(self, mapping: dict[str, Any]) -> str:
        summaries = []
        for key, value in mapping.items():
            summaries.append(f"{key}: {self._summarize_text(str(value), limit=90)}")
        return " | ".join(summaries)

    def _summarize_text(self, text: str, *, limit: int = 160) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

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

    def _retry_with_reflection(
        self,
        *,
        harness: MaterializedHarness,
        case: TaskCase,
        memory: MemoryStore,
        outputs: dict[str, str],
        traces: list[dict[str, Any]],
        trace_sink: Callable[[dict[str, Any]], None] | None,
        intra: IntraTestAdapter,
        step: WorkflowStep,
        role: RoleSpec,
        step_input: dict[str, Any],
        output_key: str,
        original_output: str,
        first_gate_trace: dict[str, Any],
    ) -> str | None:
        gate_trace = first_gate_trace
        gate_failure = f"Required quality gate failed: {gate_trace.get('gate', 'quality_gate')}"
        baseline_missing = len(gate_trace.get("missing") or [])
        previous_output = original_output
        for attempt_number in range(1, intra.max_retries(harness.genome) + 1):
            reflection_text = intra.reflection_for_failure(step=step, gate_trace=gate_trace)
            retry_input = dict(step_input)
            retry_input["temporary_reflection"] = reflection_text
            result = self.role_runner.run(role, step, retry_input, memory, harness)
            outputs[output_key] = result
            outputs[step.id] = result
            self._record_trace(
                traces,
                self._workflow_step_trace(
                    role=role,
                    step=step,
                    step_input=retry_input,
                    output_key=output_key,
                    output=result,
                    attempts=attempt_number + 1,
                )
                | {"intra_reflection_retry": True},
                trace_sink,
            )
            retry_gate_traces, retry_failure = self._run_quality_gates(harness, outputs)
            for trace in retry_gate_traces:
                self._record_trace(traces, trace | {"intra_reflection_retry": True}, trace_sink)
            latest_trace = retry_gate_traces[-1] if retry_gate_traces else gate_trace
            latest_missing = len(latest_trace.get("missing") or [])
            improved = retry_failure is None or latest_missing < baseline_missing or result != previous_output
            intra.record(
                task_id=case.id,
                step_name=step.id,
                quality_gate_name=str(gate_trace.get("gate") or "quality_gate"),
                attempt_number=attempt_number,
                reflection_text=reflection_text,
                retry_improved_output=improved,
            )
            if retry_failure is None:
                return None
            gate_trace = latest_trace
            gate_failure = retry_failure
            previous_output = result
        return gate_failure

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
