from __future__ import annotations

import shutil
import sys
from typing import Any, Callable, TextIO

import typer


def format_execution_trace(traces: list[dict]) -> str:
    lines = ["## Execution Trace"]
    for trace in traces:
        event_text = format_trace_event(trace)
        if event_text:
            lines.extend(event_text.splitlines())
    return "\n".join(lines)


def format_execution_transcript(traces: list[dict]) -> str:
    lines = ["## Agent Transcript"]
    for trace in traces:
        event_text = format_transcript_event(trace)
        if event_text:
            lines.extend(event_text.splitlines())
            lines.append("")
    if lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def echo_training_transcript_event(event: dict) -> None:
    event_text = format_training_transcript_event(event)
    if event_text:
        typer.echo(event_text)
        typer.echo("")


def compose_event_sinks(
    sinks: list[Callable[[dict[str, Any]], None]],
) -> Callable[[dict[str, Any]], None] | None:
    active = [sink for sink in sinks if sink is not None]
    if not active:
        return None

    def _sink(event: dict[str, Any]) -> None:
        for sink in active:
            sink(event)

    return _sink


class TrainingTerminalReporter:
    def __init__(
        self,
        total_generations: int,
        *,
        stream_transcript: bool = False,
        stream: TextIO | None = None,
        force_ansi: bool | None = None,
    ) -> None:
        self.total_generations = max(total_generations, 1)
        self.stream_transcript = stream_transcript
        self.stream = stream or sys.stdout
        self.use_ansi = self.stream.isatty() if force_ansi is None else force_ansi
        self.completed = 0
        self.phase = "starting"
        self.current_score: float | None = None
        self.best_score: float | None = None
        self._lines_below_status = 1
        self._write_initial_status()

    def on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "generation_start":
            generation = int(event.get("generation", 0))
            self.completed = min(generation, self.total_generations)
            self.phase = f"gen {generation} starting"
        elif kind == "evaluation_complete" and event.get("label") == "current":
            generation = int(event.get("generation", 0))
            self.completed = min(generation + 1, self.total_generations)
            self.phase = f"gen {generation} evaluated"
            self.current_score = float(event.get("score", 0.0))
            self.best_score = max(
                self.best_score if self.best_score is not None else self.current_score,
                self.current_score,
            )
        elif kind == "evaluation_start":
            label = event.get("label", "evaluation")
            generation = event.get("generation")
            self.phase = f"gen {generation} {label} eval"
        elif kind == "mutation_plan":
            self.phase = f"gen {event.get('generation')} planning"
        elif kind == "mutation_proposed":
            self.phase = f"gen {event.get('generation')} mutation {event.get('index')}"
        elif kind in {"mutation_rejected", "mutation_rolled_back"}:
            self.phase = f"gen {event.get('generation')} mutation {event.get('index')} rejected"
        elif kind == "mutation_promoted":
            self.phase = f"gen {event.get('generation')} mutation {event.get('index')} promoted"
            self.current_score = float(event.get("new_score", 0.0))
            self.best_score = max(
                self.best_score if self.best_score is not None else self.current_score,
                self.current_score,
            )
        elif kind == "hidden_evaluation_start":
            self.phase = f"gen {event.get('generation')} hidden eval"
        elif kind == "generation_complete":
            generation = int(event.get("generation", 0))
            self.completed = min(generation + 1, self.total_generations)
            self.phase = f"gen {generation} complete"
        elif kind == "freeze":
            self.completed = self.total_generations
            self.phase = "frozen"
            self.current_score = float(event.get("score", 0.0))
            self.best_score = max(
                self.best_score if self.best_score is not None else self.current_score,
                self.current_score,
            )

        self._render_status()
        detail = format_training_detail_event(event)
        if self.stream_transcript and event.get("event") == "harness_trace":
            detail = format_training_transcript_event(event)
        self._write_detail(detail)

    def finish(self, status: str) -> None:
        if self.completed < self.total_generations and status == "FROZEN":
            self.completed = self.total_generations
        status_lower = status.lower()
        finish_labels = {
            "frozen": "frozen",
            "stopped": "stopped",
            "converged": "converged",
            "budget_exceeded": "budget exceeded",
        }
        self.phase = finish_labels.get(status_lower, status_lower)
        self._render_status()
        self.stream.write("\n")
        self.stream.flush()

    def _write_initial_status(self) -> None:
        self.stream.write(self._status_line() + "\n")
        self.stream.flush()

    def _render_status(self) -> None:
        line = self._status_line()
        if self.use_ansi:
            self.stream.write(
                f"\033[{self._lines_below_status}A"
                "\r\033[2K"
                f"{line}"
                f"\033[{self._lines_below_status}B"
                "\r"
            )
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()

    def _write_detail(self, detail: str) -> None:
        if not detail:
            return
        text = detail.rstrip() + "\n"
        self.stream.write(text)
        if self.use_ansi:
            self._lines_below_status += text.count("\n")
        self.stream.flush()

    def _status_line(self) -> str:
        width = 28
        ratio = self.completed / self.total_generations
        filled = int(ratio * width)
        bar = "#" * filled + "-" * (width - filled)
        score = "score n/a" if self.current_score is None else f"score {self.current_score:.4f}"
        best = "best n/a" if self.best_score is None else f"best {self.best_score:.4f}"
        line = (
            f"🧬 Training [{bar}] {self.completed}/{self.total_generations} | "
            f"{self.phase} | {score} | {best}"
        )
        return self._fit_to_terminal(line)

    def _fit_to_terminal(self, line: str) -> str:
        columns = shutil.get_terminal_size((120, 20)).columns
        if len(line) <= columns:
            return line
        return line[: max(columns - 1, 1)]


def echo_trace_event(trace: dict) -> None:
    event_text = format_trace_event(trace)
    if event_text:
        typer.echo(event_text)


def echo_transcript_event(trace: dict) -> None:
    event_text = format_transcript_event(trace)
    if event_text:
        typer.echo(event_text)
        typer.echo("")


def format_trace_event(trace: dict) -> str:
    event = trace.get("event")
    if event == "workflow_step":
        observation = trace.get("agent_observation", {})
        return "\n".join(
            [
                f"- Role: {trace.get('role', 'unknown')}",
                f"  Step: {trace.get('step_id', 'unknown')}",
                f"  Goal: {observation.get('goal') or trace.get('step_action', '')}",
                f"  Context: {observation.get('available_context', '')}",
                "  Output: "
                f"{observation.get('output_summary') or trace.get('output_summary', '')}",
                f"  Boundary: {observation.get('reasoning_boundary', '')}",
            ]
        )
    if event == "quality_gate":
        missing = trace.get("missing") or []
        missing_text = ", ".join(str(item) for item in missing) if missing else "none"
        return "\n".join(
            [
                f"- Quality gate: {trace.get('gate', 'unknown')}",
                f"  Passed: {trace.get('passed')}",
                f"  Missing: {missing_text}",
            ]
        )
    if event == "environment_materialized":
        artifacts = ", ".join(str(item) for item in trace.get("artifacts", []))
        return f"- Environment artifacts: {artifacts}"
    return ""


def format_training_transcript_event(event: dict) -> str:
    event_type = event.get("event")
    generation = event.get("generation")
    prefix = _training_prefix(generation)
    if event_type == "evolution_start":
        mode = "resume" if event.get("resume") else "new run"
        return "\n".join(
            [
                "[Training / start]",
                f"Scenario: {event.get('scenario')}",
                f"Run: {event.get('run_id')} ({mode})",
            ]
        )
    if event_type == "generation_start":
        return "\n".join(
            [
                f"{prefix} / generation_start]",
                f"Genome version: {event.get('genome_version')}",
            ]
        )
    if event_type == "evaluation_start":
        return "\n".join(
            [
                f"{prefix} / {event.get('label')}_evaluation]",
                f"Genome version: {event.get('genome_version')}",
                _mutation_context(event),
            ]
        ).rstrip()
    if event_type == "harness_trace":
        trace_text = format_transcript_event(event.get("trace", {}))
        if not trace_text:
            return ""
        case_header = (
            f"{prefix} / {event.get('label')} / {event.get('split')}:{event.get('case_id')}]"
        )
        mutation = _mutation_context(event)
        body = [case_header]
        if mutation:
            body.append(mutation)
        body.extend(trace_text.splitlines())
        return "\n".join(body)
    if event_type == "evaluation_complete":
        validation_score = event.get("validation_score")
        validation_text = (
            "none" if validation_score is None else f"{float(validation_score):.4f}"
        )
        return "\n".join(
            [
                f"{prefix} / evaluation_complete]",
                f"Label: {event.get('label')}",
                f"Promotion score: {float(event.get('score', 0.0)):.4f}",
                f"Train score: {float(event.get('train_score', 0.0)):.4f}",
                f"Validation score: {validation_text}",
            ]
        )
    if event_type == "mutation_plan":
        failures = event.get("failure_patterns") or []
        failure_text = "; ".join(str(item) for item in failures) if failures else "none"
        return "\n".join(
            [
                f"{prefix} / Nucleus]",
                f"Plan: {event.get('summary')}",
                f"Proposed mutations: {event.get('proposed_count')}",
                f"Observed failure patterns: {failure_text}",
            ]
        )
    if event_type == "mutation_proposed":
        return "\n".join(
            [
                f"💡 {prefix} / Nucleus proposes mutation {event.get('index')}]",
                f"Type: {event.get('mutation_type')} -> {event.get('target')}",
                f"Why: {event.get('rationale')}",
                f"Expected improvement: {event.get('expected_improvement')}",
                f"Risk: {event.get('risk')}",
            ]
        )
    if event_type == "mutation_rejected":
        return "\n".join(
            [
                f"❌ {prefix} / Guardian rejects mutation {event.get('index')}]",
                f"Type: {event.get('mutation_type')} -> {event.get('target')}",
                f"Reason: {event.get('reason')}",
            ]
        )
    if event_type == "mutation_promoted":
        return "\n".join(
            [
                f"✅ {prefix} / Guardian promotes mutation {event.get('index')}]",
                f"Type: {event.get('mutation_type')} -> {event.get('target')}",
                f"Score: {float(event.get('old_score', 0.0)):.4f} -> "
                f"{float(event.get('new_score', 0.0)):.4f}",
                f"Why: {event.get('rationale')}",
            ]
        )
    if event_type == "mutation_rolled_back":
        return "\n".join(
            [
                f"↩️  {prefix} / Guardian rolls back mutation {event.get('index')}]",
                f"Type: {event.get('mutation_type')} -> {event.get('target')}",
                f"Score: {float(event.get('old_score', 0.0)):.4f} -> "
                f"{float(event.get('new_score', 0.0)):.4f}",
                f"Reason: {event.get('reason')}",
            ]
        )
    if event_type == "freeze_recommended":
        return "\n".join(
            [
                f"{prefix} / Nucleus]",
                f"Recommended freeze: {event.get('reason')}",
            ]
        )
    if event_type == "freeze":
        return "\n".join(
            [
                f"❄️  {prefix} / freeze]",
                f"Frozen score: {float(event.get('score', 0.0)):.4f}",
                f"Reason: {event.get('reason')}",
            ]
        )
    return ""


def format_training_detail_event(event: dict) -> str:
    event_type = event.get("event")
    generation = event.get("generation")
    if event_type == "evolution_start":
        mode = "resume" if event.get("resume") else "new run"
        hidden_cases = int(event.get("hidden_case_count", 0) or 0)
        hidden_text = f"{hidden_cases} cases" if hidden_cases else "not configured"
        return "\n".join(
            [
                "=== stem_agent evolution ===",
                f"Scenario: {event.get('scenario')} ({event.get('task_class')})",
                f"Scenario path: {event.get('scenario_path')}",
                f"Run id: {event.get('run_id')} ({mode})",
                f"Run directory: {event.get('run_dir')}",
                f"Model: {event.get('model')}",
                f"Endpoint: {event.get('endpoint')}",
                "",
                "Configuration overview:",
                f"  train cases: {event.get('train_case_count')}",
                f"  validation cases: {event.get('validation_case_count')}",
                f"  hidden evaluation: {hidden_text}",
                f"  max generations: {event.get('max_generations')}",
                f"  max mutations/generation: {event.get('max_mutations_per_generation')}",
                f"  patience: {event.get('patience')}",
                f"  min promotion delta: {event.get('min_delta')}",
                f"  max cost: ${float(event.get('max_cost_usd', 0.0)):.2f}",
            ]
        )
    if event_type == "generation_start":
        return "\n".join(
            [
                f"=== Generation {generation} ===",
                f"Genome version: {event.get('genome_version')}",
            ]
        )
    if event_type == "diagnosis_complete":
        constraints = event.get("constraint_count")
        requirements = event.get("requirement_count")
        return "\n".join(
            [
                "Nucleus diagnosis complete",
                f"  Initial hypothesis: {event.get('initial_evaluation_hypothesis')}",
                f"  Requirements: {requirements}",
                f"  Constraints: {constraints}",
            ]
        )
    if event_type == "evaluation_start":
        label = event.get("label")
        mutation = _mutation_context(event)
        lines = [
            f"Evaluation starting: {label}",
            f"  generation: {generation}",
            f"  genome version: {event.get('genome_version')}",
            f"  workspace: {event.get('workspace_dir')}",
            f"  train cases: {event.get('train_case_count')}",
            f"  validation cases: {event.get('validation_case_count')}",
        ]
        if mutation:
            lines.append(f"  {mutation}")
        return "\n".join(lines)
    if event_type == "case_start":
        return f"▶ Case started  | {_case_context(event)}"
    if event_type == "case_complete":
        return "\n".join(
            [
                f"✅ Case complete | {_case_context(event)}",
                f"   preview: {_clean_output_preview(event.get('output_summary'))}",
            ]
        )
    if event_type == "evaluation_artifacts_written":
        return "\n".join(
            [
                f"💾 Artifacts     | label={event.get('label')}",
                f"   outputs: {event.get('outputs_dir')}",
                f"   traces:  {event.get('traces_path')}",
            ]
        )
    if event_type == "evaluation_complete":
        validation_score = event.get("validation_score")
        validation_text = (
            "none" if validation_score is None else f"{float(validation_score):.4f}"
        )
        return "\n".join(
            [
                f"Evaluation complete: {event.get('label')}",
                f"  promotion score: {float(event.get('score', 0.0)):.4f}",
                f"  train score: {float(event.get('train_score', 0.0)):.4f}",
                f"  validation score: {validation_text}",
            ]
        )
    if event_type == "hidden_evaluation_start":
        return "\n".join(
            [
                "Hidden evaluation starting",
                f"  generation: {generation}",
                f"  genome id: {event.get('genome_id')}",
                f"  workspace: {event.get('workspace_dir')}",
                f"  label: {event.get('label')}",
            ]
        )
    if event_type == "hidden_evaluation_complete":
        return "\n".join(
            [
                "Hidden evaluation complete",
                f"  generation: {generation}",
                f"  hidden score: {float(event.get('hidden_score', 0.0)):.4f}",
                f"  label: {event.get('label')}",
            ]
        )
    if event_type == "mutation_plan":
        failures = event.get("failure_patterns") or []
        failure_text = "; ".join(str(item) for item in failures) if failures else "none"
        return "\n".join(
            [
                "Nucleus mutation plan",
                f"  operator: {event.get('operator_type')}",
                f"  summary: {event.get('summary')}",
                f"  proposed mutations: {event.get('proposed_count')}",
                f"  observed failure patterns: {failure_text}",
            ]
        )
    if event_type == "mutation_proposed":
        return "\n".join(
            [
                f"💡 Mutation proposed: #{event.get('index')}",
                f"  type: {event.get('mutation_type')} -> {event.get('target')}",
                f"  operator: {event.get('operator_type')}",
                f"  expected improvement: {event.get('expected_improvement')}",
                f"  risk: {event.get('risk')}",
            ]
        )
    if event_type == "mutation_rejected":
        reviewer = event.get("mutation_rejected_by") or "guardian"
        return "\n".join(
            [
                f"❌ Mutation rejected: #{event.get('index')}",
                f"  type: {event.get('mutation_type')} -> {event.get('target')}",
                f"  reviewer: {reviewer}",
                f"  reason: {event.get('reason')}",
            ]
        )
    if event_type == "mutation_promoted":
        hidden = "yes" if event.get("hidden_eval_regression") else "no"
        return "\n".join(
            [
                f"✅ Mutation promoted: #{event.get('index')}",
                f"  type: {event.get('mutation_type')} -> {event.get('target')}",
                f"  score: {float(event.get('old_score', 0.0)):.4f} -> "
                f"{float(event.get('new_score', 0.0)):.4f}",
                f"  hidden regression: {hidden}",
                f"  rationale: {event.get('rationale')}",
            ]
        )
    if event_type == "mutation_rolled_back":
        return "\n".join(
            [
                f"↩️  Mutation rolled back: #{event.get('index')}",
                f"  type: {event.get('mutation_type')} -> {event.get('target')}",
                f"  score: {float(event.get('old_score', 0.0)):.4f} -> "
                f"{float(event.get('new_score', 0.0)):.4f}",
                f"  reason: {event.get('reason')}",
            ]
        )
    if event_type == "convergence_decision":
        return "\n".join(
            [
                "⚖️  Convergence decision",
                f"  action: {event.get('action')}",
                f"  stop: {event.get('stop')}",
                f"  reason: {event.get('reason')}",
                f"  plateau detected: {event.get('plateau_detected')}",
            ]
        )
    if event_type == "generation_complete":
        return "\n".join(
            [
                f"Generation {generation} complete",
                f"  best score: {float(event.get('best_score', 0.0)):.4f}",
                f"  patience left: {event.get('patience_left')}",
                f"  tokens used this generation: {event.get('tokens_used')}",
            ]
        )
    if event_type == "freeze_recommended":
        return f"Nucleus recommended freeze: {event.get('reason')}"
    if event_type == "freeze":
        return "\n".join(
            [
                "❄️  Final freeze complete",
                f"  generation: {generation}",
                f"  frozen score: {float(event.get('score', 0.0)):.4f}",
                f"  reason: {event.get('reason')}",
            ]
        )
    return ""


def _training_prefix(generation: object) -> str:
    if generation is None:
        return "[Training"
    return f"[Training / generation {generation}"


def _mutation_context(event: dict) -> str:
    mutation_index = event.get("mutation_index")
    if mutation_index is None:
        return ""
    return f"Mutation candidate: {mutation_index}"


def _case_context(event: dict) -> str:
    return (
        f"label={event.get('label')} | "
        f"split={event.get('split')} | "
        f"case={event.get('case_id')}"
    )


def _clean_output_preview(summary: object) -> str:
    text = " ".join(str(summary or "").split())
    for marker in ("```markdown", "```"):
        text = text.replace(marker, "").strip()
    return text or "empty output"


def format_transcript_event(trace: dict) -> str:
    event = trace.get("event")
    if event == "environment_materialized":
        artifacts = ", ".join(str(item) for item in trace.get("artifacts", []))
        return "\n".join(
            [
                "[Harness / environment]",
                f"Prepared workspace artifacts: {artifacts}",
            ]
        )
    if event == "workflow_step":
        observation = trace.get("agent_observation", {})
        role = trace.get("role", "unknown")
        step_id = trace.get("step_id", "unknown")
        goal = observation.get("goal") or trace.get("step_action", "")
        context = observation.get("available_context", "")
        output = observation.get("output_summary") or trace.get("output_summary", "")
        return "\n".join(
            [
                f"[{role} / {step_id}]",
                f"Goal: {goal}",
                f"I can see: {context}",
                f"I produced: {output}",
                "Note: this is visible execution state, not hidden chain-of-thought.",
            ]
        )
    if event == "quality_gate":
        missing = trace.get("missing") or []
        missing_text = ", ".join(str(item) for item in missing) if missing else "none"
        return "\n".join(
            [
                f"[Guardian / {trace.get('gate', 'quality_gate')}]",
                f"Passed: {trace.get('passed')}",
                f"Missing requirements: {missing_text}",
            ]
        )
    return ""
