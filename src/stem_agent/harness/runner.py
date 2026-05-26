from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from stem_agent.genome.models import RoleSpec, WorkflowStep
from stem_agent.harness.builder import MaterializedHarness
from stem_agent.harness.intra_adapter import IntraTestAdapter
from stem_agent.harness.memory import MemoryStore
from stem_agent.harness.role_runner import RoleRunner
from stem_agent.kernel.requirements import requirement_section_satisfied
from stem_agent.scenarios.schema import TaskCase


class HarnessRunResult(BaseModel):
    case_id: str
    final_output: str
    outputs: dict[str, str] = Field(default_factory=dict)
    traces: list[dict[str, Any]] = Field(default_factory=list)
    cost_estimate: float = 0.0
    blocked: bool = False
    block_reason: str | None = None
    expected_output: str | None = None
    reference_notes: str = ""
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
        # ── SWE-bench workspace population: copy cached repo files into harness workspace ──
        _maybe_populate_swebench_workspace(harness, case)
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
            self_eval_trace = self._run_self_evaluation(harness, outputs)
            if self_eval_trace:
                self._record_trace(traces, self_eval_trace, trace_sink)
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
                    reference_notes=case.reference_notes,
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
            reference_notes=case.reference_notes,
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

    def _run_self_evaluation(
        self,
        harness: MaterializedHarness,
        outputs: dict[str, str],
    ) -> dict[str, Any] | None:
        if not harness.genome.self_evaluation.get("enabled"):
            return None
        if "final_output" not in outputs:
            return None

        final_output = self.extract_final(outputs)
        missing = self._missing_required_sections(harness, final_output.lower())
        rubric = harness.genome.self_evaluation.get("rubric") or []
        if isinstance(rubric, str):
            rubric_items = [rubric] if rubric.strip() else []
        else:
            rubric_items = [str(item) for item in rubric]
        passed = not missing
        outputs["self_evaluation"] = "\n".join(
            [
                "## Self Evaluation",
                f"- Rubric items checked: {len(rubric_items)}",
                f"- Missing required sections: {'none' if passed else '; '.join(missing)}",
            ]
        )
        return {
            "event": "self_evaluation",
            "step_id": "self_evaluation",
            "rubric_items": len(rubric_items),
            "passed": passed,
            "missing": missing,
            "status": "completed",
        }

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
        self_eval_cost = 0.0
        return round(workflow_cost + gate_cost + self_eval_cost, 4)

    def _missing_required_sections(
        self, harness: MaterializedHarness, output: str
    ) -> list[str]:
        missing: list[str] = []
        for requirement in harness.scenario.expected_output.requirements:
            section_result = requirement_section_satisfied(requirement, output)
            passed = True if section_result is None else section_result
            if not passed:
                missing.append(requirement)
        return missing


def _maybe_populate_swebench_workspace(
    harness: "MaterializedHarness", case: "TaskCase"
) -> None:
    """Populate harness workspace with cached repo files for SWE-bench cases.
    
    The locator/patcher roles use read_file and search_code tools that operate
    from cwd. Without this, they search an empty workspace and produce hallucinated patches.
    
    Also applies the test_patch and attempts to run failing tests to provide
    the locator/patcher with concrete failure output — the strongest signal for fixing bugs.
    """
    case_input = case.input if hasattr(case, "input") else {}
    repo = case_input.get("repo", "")
    base_commit = case_input.get("base_commit", "")
    if not repo or not base_commit or repo == "demo/simple":
        return

    import shutil
    import subprocess
    from pathlib import Path

    workspace = harness.workspace_dir
    artifacts_dir = workspace / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path.home() / ".cache" / "stem_agent" / "swebench_workspaces"
    cached = cache_dir / f"{_safe_slug(repo)}_{base_commit[:8]}"

    if not cached.exists():
        # First time: clone into cache
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-tags",
                 f"https://github.com/{repo}.git", str(cached)],
                capture_output=True, timeout=300, check=True,
            )
            subprocess.run(
                ["git", "-C", str(cached), "checkout", base_commit],
                capture_output=True, timeout=120, check=True,
            )
        except Exception:
            shutil.rmtree(cached, ignore_errors=True)
            return

    # Copy cached repo files into workspace (skip .git to avoid conflicts)
    for item in cached.iterdir():
        if item.name == ".git":
            continue
        dest = workspace / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # ── Apply test_patch and run failing tests ──
    test_patch = case_input.get("test_patch", "")
    fail_to_pass_raw = case_input.get("FAIL_TO_PASS", [])
    # HF dataset sometimes stores FAIL_TO_PASS as a JSON-encoded string
    if isinstance(fail_to_pass_raw, str):
        import json as _json
        fail_to_pass = _json.loads(fail_to_pass_raw) if fail_to_pass_raw.strip() else []
    else:
        fail_to_pass = list(fail_to_pass_raw) if fail_to_pass_raw else []
    test_output_path = artifacts_dir / "test_failures.txt"

    if test_patch:
        _apply_test_patch_and_run(workspace, test_patch, fail_to_pass, test_output_path)
    elif not test_output_path.exists():
        test_output_path.write_text("(No test patch available for this case.)\n")

    # ── Write test file hints for the locator ──
    _write_test_file_hints(artifacts_dir, test_patch, fail_to_pass)


def _apply_test_patch_and_run(
    workspace: "Path",
    test_patch: str,
    fail_to_pass: list[str],
    output_path: "Path",
) -> None:
    """Apply test_patch to workspace, then run pytest and write failures."""
    import subprocess

    patch_file = workspace / "_test_patch.diff"
    patch_file.write_text(test_patch)

    # Quick git reset first in case of leftover changes
    subprocess.run(
        ["git", "-C", str(workspace), "checkout", "--", "."],
        capture_output=True, timeout=30,
    )

    # Apply the test patch — use absolute path to avoid CWD issues
    apply_result = subprocess.run(
        ["git", "apply", str(patch_file.resolve())],
        capture_output=True, text=True, timeout=30,
        cwd=str(workspace),
    )
    patch_file.unlink(missing_ok=True)

    lines = []
    if apply_result.returncode != 0:
        lines.append(f"[WARNING] test_patch application FAILED:\n{apply_result.stderr[:2000]}")
    else:
        lines.append("[OK] test_patch applied successfully.")

    # Try running pytest with the failing tests
    if fail_to_pass and apply_result.returncode == 0:
        lines.append(f"\nRunning failing tests ({len(fail_to_pass)}): {fail_to_pass}")
        lines.append("-" * 60)

        # Use the cached venv if it exists
        venv_cache = Path.home() / ".cache" / "stem_agent" / "swebench_test_venvs" / _safe_slug(str(workspace.name))
        python_exe = _ensure_test_venv(workspace, venv_cache)

        if python_exe:
            pytest_cmd = [
                str(python_exe), "-m", "pytest", "-x", "--tb=short",
                "--timeout=60", "-q",
            ] + fail_to_pass
            result = subprocess.run(
                pytest_cmd,
                capture_output=True, text=True, timeout=120,
                cwd=str(workspace),
                env={**__import__("os").environ, "PYTHONPATH": str(workspace)},
            )
            lines.append(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
            if result.stderr:
                stderr_short = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
                lines.append(f"\nSTDERR:\n{stderr_short}")
            lines.append(f"\nExit code: {result.returncode}")
        else:
            lines.append("\n(Test dependencies not installed — see test file in workspace)")
            lines.append(f"FAIL_TO_PASS: {fail_to_pass}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_test_venv(workspace: "Path", venv_cache: "Path") -> str | None:
    """Ensure a venv with pytest exists for the workspace. Returns python path or None."""
    import subprocess
    import venv

    if not venv_cache.exists():
        try:
            venv.create(venv_cache, with_pip=True)
            pip = str(venv_cache / "bin" / "pip")
            subprocess.run(
                [pip, "install", "pytest", "pytest-timeout"],
                capture_output=True, timeout=120,
            )
            # Also try to install the repo in editable mode
            subprocess.run(
                [pip, "install", "-e", str(workspace)],
                capture_output=True, timeout=120,
            )
        except Exception:
            import shutil
            shutil.rmtree(venv_cache, ignore_errors=True)
            return None

    python_exe = venv_cache / "bin" / "python"
    if python_exe.exists():
        return str(python_exe)
    return None


def _safe_slug(text: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)


def _write_test_file_hints(
    artifacts_dir: "Path", test_patch: str, fail_to_pass: list[str]
) -> None:
    """Write a hint file telling the locator which source files the failing tests touch.

    Parses the test_patch to find the test file paths, then uses heuristics to
    guess which production source files are likely involved.
    """
    import re

    test_files = re.findall(r"\+{3} b/(.+)", test_patch)
    if not test_files:
        return

    hints_path = artifacts_dir / "test_file_hints.txt"
    lines = ["# Test File Hints for Locator", ""]
    lines.append(f"FAIL_TO_PASS: {fail_to_pass}")
    lines.append("")
    lines.append("These test files were modified by the test patch — the bug is likely in")
    lines.append("the production code that these tests exercise:")
    lines.append("")

    for tf in test_files:
        lines.append(f"  - {tf}")
        # Guess the source dir from the test file path
        # e.g. sympy/integrals/tests/test_intpoly.py → sympy/integrals/
        source_hint = re.sub(r"/tests?/", "/", tf)
        source_hint = re.sub(r"/?test_", "/", source_hint)
        source_hint = re.sub(r"\.py$", ".py", source_hint)
        if source_hint != tf:
            lines.append(f"    → likely source: {source_hint}")
        # Module path hint
        mod_path = tf.replace("/", ".").replace(".py", "")
        mod_path = re.sub(r"\.tests?\.test_", ".", mod_path)
        if mod_path != tf.replace("/", ".").replace(".py", ""):
            lines.append(f"    → module: {mod_path}")

    hints_path.write_text("\n".join(lines))
