from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import typer
import yaml

from stem_agent.evolution.control import EvolutionControl
from stem_agent.evolution.loop import EvolutionLoop
from stem_agent.evolution.visuals import VisualizationBuilder
from stem_agent.benchmarks.gsm8k import download_gsm8k_sample
from stem_agent.cli_compare_ablations import compare_ablations as compare_ablations_report
from stem_agent.genome.loader import load_genome
from stem_agent.harness.builder import HarnessBuilder
from stem_agent.harness.runner import HarnessRunner
from stem_agent.kernel.versioning import GitProvenance, GitRunConfig
from stem_agent.kernel.signal_audit import SignalLeakAuditor, nucleus_visible_artifact_paths
from stem_agent.kernel.versioning import GenomeArchive
from stem_agent.scenarios.loader import load_scenario
from stem_agent.scenarios.schema import TaskCase

app = typer.Typer(help="stem_agent command line interface.")


@app.command("init-scenario")
def init_scenario(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    scenario = {
        "scenario": {
            "name": path.name,
            "description": "Describe the class of tasks this harness should learn.",
            "task_class": "general task class",
        },
        "input_format": {
            "type": "text",
            "fields": [{"name": "user_request", "required": True}],
        },
        "expected_output": {
            "type": "markdown",
            "requirements": [
                "Must include a short summary",
                "Must include concrete steps",
                "Must include final answer",
            ],
        },
        "constraints": ["Prefer actionable output"],
        "available_builtin_tools": ["call_model"],
        "success_criteria": [
            {
                "name": "requirement_coverage",
                "weight": 0.5,
                "description": "Output satisfies required sections.",
            },
            {
                "name": "usefulness",
                "weight": 0.5,
                "description": "Output is useful and actionable.",
            },
        ],
        "signal_policy": {
            "layer_1_enabled": True,
            "expose_aggregate_score": True,
            "expose_score_delta": True,
            "expose_direction": True,
        },
        "evolution": {
            "max_generations": 5,
            "patience": 2,
            "min_delta": 0.03,
            "max_mutations_per_generation": 4,
            "max_cost_usd": 2.0,
        },
    }
    (path / "scenario.yaml").write_text(
        yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8"
    )
    (path / "train_cases.jsonl").write_text(
        json.dumps(
            {
                "id": "train_001",
                "input": {"user_request": "Describe one representative training task."},
                "reference_notes": "Should satisfy the expected output requirements.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "validation_cases.jsonl").write_text(
        json.dumps(
            {
                "id": "val_001",
                "input": {"user_request": "Describe one representative validation task."},
                "reference_notes": "Should validate generalization, not memorization.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Initialized scenario at {path}")


@app.command("init-benchmark")
def init_benchmark(
    name: str,
    n_train: int = typer.Option(30, "--n-train"),
    n_val: int = typer.Option(20, "--n-val"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    if name != "gsm8k_mini":
        raise typer.BadParameter("Supported benchmark: gsm8k_mini")
    path = Path("scenarios") / name
    path.mkdir(parents=True, exist_ok=True)
    all_cases, _ = download_gsm8k_sample(n_train=n_train + n_val + 10, n_val=0, seed=seed)
    train_cases = [dict(item, id=f"train_{index + 1:03d}") for index, item in enumerate(all_cases[:n_train])]
    val_cases = [
        dict(item, id=f"val_{index + 1:03d}")
        for index, item in enumerate(all_cases[n_train : n_train + n_val])
    ]
    hidden_cases = [
        dict(item, id=f"hidden_{index + 1:03d}")
        for index, item in enumerate(all_cases[n_train + n_val : n_train + n_val + 10])
    ]
    train_inputs = {json.dumps(item["input"], sort_keys=True) for item in train_cases}
    val_inputs = {json.dumps(item["input"], sort_keys=True) for item in val_cases}
    if train_inputs & val_inputs:
        raise RuntimeError("GSM8K train/validation split overlap detected")
    scenario = _gsm8k_scenario_yaml()
    (path / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
    _write_jsonl(path / "train_cases.jsonl", train_cases)
    _write_jsonl(path / "validation_cases.jsonl", val_cases)
    _write_jsonl(path / "hidden_cases.jsonl", hidden_cases)
    typer.echo(f"Initialized benchmark at {path}")


@app.command()
def evolve(
    scenario_path: Path,
    run_id: str = typer.Option(..., "--run-id"),
    git_branch: bool = typer.Option(False, "--git-branch"),
    git_commit: bool = typer.Option(False, "--git-commit"),
    git_push: bool = typer.Option(False, "--git-push"),
    progress_bar: bool = typer.Option(
        True,
        "--progress-bar/--no-progress-bar",
        help="Show a live generation progress bar during evolution.",
    ),
    stream_training_transcript: bool = typer.Option(
        False,
        "--stream-training-transcript",
        help="Print agent outputs and Nucleus/Guardian decisions during evolution.",
    ),
    signal_policy_override: Optional[list[str]] = typer.Option(
        None,
        "--signal-policy-override",
        help="Override signal_policy fields, e.g. layer_1_enabled=false.",
    ),
) -> None:
    scenario = load_scenario(scenario_path).scenario
    sinks: list[Callable[[dict[str, Any]], None]] = []
    progress = None
    if progress_bar:
        progress = _TrainingTerminalReporter(
            total_generations=max(int(scenario.convergence_policy.max_generations), 1),
            stream_transcript=stream_training_transcript,
        )
        sinks.append(progress.on_event)
    if stream_training_transcript and progress is None:
        sinks.append(_echo_training_transcript_event)
    event_sink = _compose_event_sinks(sinks)
    result = None
    try:
        result = EvolutionLoop().evolve(
            scenario_path,
            run_id,
            git_branch=git_branch,
            git_commit=git_commit,
            git_push=git_push,
            signal_policy_override=_parse_signal_policy_overrides(signal_policy_override or []),
            event_sink=event_sink,
        )
    finally:
        if progress:
            progress.finish(status=result.status if result else "stopped")
    typer.echo(f"Run directory: {result.run_dir}")
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Baseline score: {result.baseline_score:.4f}")
    typer.echo(f"Final score: {result.final_score:.4f}")
    typer.echo(f"Frozen genome: {result.frozen_genome_path}")
    typer.echo(f"Report: {result.report_path}")


@app.command()
def inspect(run_path: Path) -> None:
    lineage_path = run_path / "lineage.jsonl"
    report_path = run_path / "report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        if "## Genome Archive" not in report:
            report += "\n\n## Genome Archive\n" + GenomeArchive(run_dir=run_path).to_markdown_table() + "\n"
        typer.echo(report)
        return
    if not lineage_path.exists():
        raise typer.BadParameter(f"Missing run artifacts at {run_path}")
    typer.echo(lineage_path.read_text(encoding="utf-8"))
    typer.echo("")
    typer.echo("## Genome Archive")
    typer.echo(GenomeArchive(run_dir=run_path).to_markdown_table())


@app.command()
def compare(run_path: Path) -> None:
    baseline = _read_eval(run_path / "generation_000" / "eval_result.json")
    final = _read_eval(run_path / "final_evaluation" / "eval_result.json")
    typer.echo(f"Baseline score: {baseline['promotion_score']:.4f}")
    baseline_score_path = run_path / "baseline_genome_score.json"
    if baseline_score_path.exists():
        baseline_score = json.loads(baseline_score_path.read_text(encoding="utf-8"))
        typer.echo(f"Baseline genome score: {float(baseline_score['score']):.4f}")
    typer.echo(f"Final score: {final['promotion_score']:.4f}")
    typer.echo(f"Improvement: {final['promotion_score'] - baseline['promotion_score']:.4f}")
    lineage_path = run_path / "lineage.jsonl"
    if lineage_path.exists():
        promoted = []
        rejected = []
        for line in lineage_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event"] == "mutation_promoted":
                promoted.append(f"{event['mutation_type']} -> {event['target']}")
            if event["event"] in {"mutation_rejected", "mutation_rolled_back"}:
                rejected.append(f"{event['event']} {event['mutation_type']} -> {event['target']}")
        typer.echo("Promoted mutations:")
        typer.echo("\n".join(f"  - {item}" for item in promoted) or "  - None")
        typer.echo("Rejected or rolled back mutations:")
        typer.echo("\n".join(f"  - {item}" for item in rejected) or "  - None")


@app.command()
def visualize(run_path: Path) -> None:
    paths = VisualizationBuilder().visualize(run_path, update_report=True)
    typer.echo(f"Visuals written to {run_path / 'visuals'}")
    for path in paths:
        typer.echo(f"- {path}")


@app.command()
def progress(run_path: Path) -> None:
    path = VisualizationBuilder().progress(run_path)
    typer.echo(f"Training progress written to {path}")


@app.command()
def aggregate(run_paths: list[Path]) -> None:
    path = VisualizationBuilder().aggregate(run_paths)
    typer.echo(f"Aggregate report: {path}")


@app.command("compare-ablations")
def compare_ablations(run_paths: list[Path]) -> None:
    report = compare_ablations_report([str(path) for path in run_paths])
    typer.echo(f"Ablation comparison written to {report}")


@app.command()
def execute(
    frozen_genome: Path,
    input: str = typer.Option(..., "--input"),
    scenario_path: Optional[Path] = typer.Option(None, "--scenario-path"),
    show_trace: bool = typer.Option(
        False,
        "--show-trace",
        help="Show observable role execution state after the final output.",
    ),
    show_transcript: bool = typer.Option(
        False,
        "--show-transcript",
        help="Show a readable agent transcript after the final output.",
    ),
    stream_trace: bool = typer.Option(
        False,
        "--stream-trace",
        help="Print observable role execution state as the harness runs.",
    ),
    stream_transcript: bool = typer.Option(
        False,
        "--stream-transcript",
        help="Print a readable agent transcript as the harness runs.",
    ),
) -> None:
    genome = load_genome(frozen_genome)
    scenario_root = scenario_path or _infer_scenario_path(frozen_genome)
    bundle = load_scenario(scenario_root)
    harness = HarnessBuilder().materialize(
        genome,
        bundle.scenario,
        workspace_dir=frozen_genome.parent / "execute_workspace",
    )
    case = TaskCase(id="execute_001", input={"user_request": input})
    trace_sink = None
    if stream_transcript:
        trace_sink = _echo_transcript_event
    elif stream_trace:
        trace_sink = _echo_trace_event
    result = HarnessRunner().run_case(harness, case, trace_sink=trace_sink)
    if stream_trace or stream_transcript:
        typer.echo("")
        typer.echo("## Final Output")
    typer.echo(result.final_output)
    if show_trace:
        typer.echo("")
        typer.echo(_format_execution_trace(result.traces))
    if show_transcript:
        typer.echo("")
        typer.echo(_format_execution_transcript(result.traces))


@app.command("pause")
def pause_run(run_id: str) -> None:
    state = EvolutionControl(Path("runs") / run_id, run_id).write_command("pause")
    typer.echo(f"Pause requested for {run_id}: {state.status}")


@app.command("resume")
def resume_run(
    run_id: str,
    git_branch: bool = typer.Option(False, "--git-branch"),
    git_commit: bool = typer.Option(False, "--git-commit"),
    git_push: bool = typer.Option(False, "--git-push"),
) -> None:
    control = EvolutionControl(Path("runs") / run_id, run_id)
    control.write_command("resume", status="REQUESTED")
    result = EvolutionLoop().resume(
        run_id,
        git_branch=git_branch,
        git_commit=git_commit,
        git_push=git_push,
    )
    typer.echo(f"Run directory: {result.run_dir}")
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Final score: {result.final_score:.4f}")


@app.command("status")
def status_run(run_id: str) -> None:
    control = EvolutionControl(Path("runs") / run_id, run_id)
    typer.echo(control.read().model_dump_json(indent=2))


@app.command("freeze-now")
def freeze_now_run(
    run_id: str,
    git_branch: bool = typer.Option(False, "--git-branch"),
    git_commit: bool = typer.Option(False, "--git-commit"),
) -> None:
    control = EvolutionControl(Path("runs") / run_id, run_id)
    control.write_command("freeze_now")
    try:
        result = EvolutionLoop().freeze_now(
            run_id,
            git_branch=git_branch,
            git_commit=git_commit,
        )
    except FileNotFoundError:
        typer.echo(f"Freeze-now requested for active run {run_id}")
        return
    typer.echo(f"Frozen genome: {result.frozen_genome_path}")
    typer.echo(f"Report: {result.report_path}")


@app.command("abort")
def abort_run(
    run_id: str,
    freeze_best: bool = typer.Option(False, "--freeze-best"),
) -> None:
    command = "freeze_now" if freeze_best else "abort"
    state = EvolutionControl(Path("runs") / run_id, run_id).write_command(
        command,
        freeze_best=freeze_best,
    )
    typer.echo(f"{command} requested for {run_id}: {state.status}")


@app.command("push-run")
def push_run(run_id: str, remote: str = typer.Option("origin", "--remote")) -> None:
    git = GitProvenance(
        GitRunConfig(
            run_id=run_id,
            run_dir=Path("runs") / run_id,
            branch_enabled=True,
            commit_enabled=False,
            push_enabled=True,
            remote=remote,
        )
    )
    result = git.start()
    if result.allowed:
        result = git.push()
    if not result.allowed:
        raise typer.BadParameter(result.reason)
    typer.echo(result.reason)


def _read_eval(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"Missing evaluation result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_signal_policy_overrides(items: list[str]) -> dict[str, bool]:
    overrides: dict[str, bool] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(
                "--signal-policy-override must use key=value, e.g. layer_1_enabled=false"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = raw_value.strip().lower()
        if key not in {
            "layer_1_enabled",
            "expose_aggregate_score",
            "expose_score_delta",
            "expose_direction",
        }:
            raise typer.BadParameter(f"Unsupported signal_policy override: {key}")
        if value not in {"true", "false"}:
            raise typer.BadParameter(f"Signal policy override must be true/false: {item}")
        overrides[key] = value == "true"
    return overrides


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gsm8k_scenario_yaml() -> dict[str, Any]:
    return {
        "scenario": {
            "name": "gsm8k_mini",
            "description": "Grade school math word problems requiring multi-step arithmetic reasoning",
            "task_class": "structured_reasoning",
        },
        "input_format": {
            "type": "text",
            "fields": [{"name": "problem", "required": True}],
        },
        "expected_output": {
            "type": "markdown",
            "requirements": [
                "Must show arithmetic reasoning steps",
                "Must include the final numeric answer",
            ],
        },
        "evaluation_criteria": [
            {
                "name": "correct_answer",
                "weight": 0.7,
                "method": "exact_match_after_extraction",
            },
            {
                "name": "reasoning_steps_present",
                "weight": 0.2,
                "method": "regex_check",
                "pattern": r"\d+.*[\+\-\*\/].*\d+",
            },
            {
                "name": "no_hallucinated_numbers",
                "weight": 0.1,
                "method": "llm_judge",
            },
        ],
        "signal_policy": {
            "layer_1_enabled": True,
            "expose_aggregate_score": True,
            "expose_score_delta": True,
            "expose_direction": True,
        },
        "constraints": ["Show concise work before the final answer."],
        "available_builtin_tools": ["call_model"],
        "success_criteria": [
            {
                "name": "correct_answer",
                "weight": 0.7,
                "description": "Extracted final answer matches the gold answer.",
            },
            {
                "name": "reasoning_steps_present",
                "weight": 0.2,
                "description": "Output includes visible arithmetic reasoning.",
            },
            {
                "name": "no_hallucinated_numbers",
                "weight": 0.1,
                "description": "Reasoning does not introduce unsupported numbers.",
            },
        ],
        "evolution": {
            "max_generations": 10,
            "patience": 4,
            "min_delta": 0.03,
            "max_mutations_per_generation": 4,
            "max_cost_usd": 2.0,
            "max_workflow_steps": 8,
            "max_roles": 6,
            "max_environment_artifacts": 8,
        },
    }


def _format_execution_trace(traces: list[dict]) -> str:
    lines = ["## Execution Trace"]
    for trace in traces:
        event_text = _format_trace_event(trace)
        if event_text:
            lines.extend(event_text.splitlines())
    return "\n".join(lines)


def _format_execution_transcript(traces: list[dict]) -> str:
    lines = ["## Agent Transcript"]
    for trace in traces:
        event_text = _format_transcript_event(trace)
        if event_text:
            lines.extend(event_text.splitlines())
            lines.append("")
    if lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _echo_training_transcript_event(event: dict) -> None:
    event_text = _format_training_transcript_event(event)
    if event_text:
        typer.echo(event_text)
        typer.echo("")


def _compose_event_sinks(
    sinks: list[Callable[[dict[str, Any]], None]],
) -> Callable[[dict[str, Any]], None] | None:
    active = [sink for sink in sinks if sink is not None]
    if not active:
        return None

    def _sink(event: dict[str, Any]) -> None:
        for sink in active:
            sink(event)

    return _sink


class _TrainingTerminalReporter:
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
            self.phase = f"generation {generation} starting"
        elif kind == "evaluation_complete" and event.get("label") == "current":
            generation = int(event.get("generation", 0))
            self.completed = min(generation + 1, self.total_generations)
            self.phase = f"generation {generation} evaluated"
            self.current_score = float(event.get("score", 0.0))
            self.best_score = max(
                self.best_score if self.best_score is not None else self.current_score,
                self.current_score,
            )
        elif kind == "evaluation_start":
            label = event.get("label", "evaluation")
            generation = event.get("generation")
            self.phase = f"generation {generation} {label} evaluation"
        elif kind == "mutation_plan":
            self.phase = f"generation {event.get('generation')} planning mutations"
        elif kind == "mutation_proposed":
            self.phase = f"generation {event.get('generation')} checking mutation {event.get('index')}"
        elif kind in {"mutation_rejected", "mutation_rolled_back"}:
            self.phase = f"generation {event.get('generation')} mutation {event.get('index')} rejected"
        elif kind == "mutation_promoted":
            self.phase = f"generation {event.get('generation')} mutation {event.get('index')} promoted"
            self.current_score = float(event.get("new_score", 0.0))
            self.best_score = max(
                self.best_score if self.best_score is not None else self.current_score,
                self.current_score,
            )
        elif kind == "hidden_evaluation_start":
            self.phase = f"generation {event.get('generation')} hidden evaluation"
        elif kind == "generation_complete":
            generation = int(event.get("generation", 0))
            self.completed = min(generation + 1, self.total_generations)
            self.phase = f"generation {generation} complete"
        elif kind == "freeze":
            self.completed = self.total_generations
            self.phase = "frozen"
            self.current_score = float(event.get("score", 0.0))
            self.best_score = max(
                self.best_score if self.best_score is not None else self.current_score,
                self.current_score,
            )

        self._render_status()
        detail = _format_training_detail_event(event)
        if self.stream_transcript and event.get("event") == "harness_trace":
            detail = _format_training_transcript_event(event)
        self._write_detail(detail)

    def finish(self, status: str) -> None:
        if self.completed < self.total_generations and status == "FROZEN":
            self.completed = self.total_generations
        self.phase = status.lower()
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
            f"Training progress [{bar}] {self.completed}/{self.total_generations} | "
            f"{self.phase} | {score} | {best}"
        )
        return self._fit_to_terminal(line)

    def _fit_to_terminal(self, line: str) -> str:
        columns = shutil.get_terminal_size((120, 20)).columns
        if len(line) <= columns:
            return line
        return line[: max(columns - 1, 1)]


def _echo_trace_event(trace: dict) -> None:
    event_text = _format_trace_event(trace)
    if event_text:
        typer.echo(event_text)


def _echo_transcript_event(trace: dict) -> None:
    event_text = _format_transcript_event(trace)
    if event_text:
        typer.echo(event_text)
        typer.echo("")


def _format_trace_event(trace: dict) -> str:
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


def _format_training_transcript_event(event: dict) -> str:
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
        trace_text = _format_transcript_event(event.get("trace", {}))
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
                f"{prefix} / Nucleus proposes mutation {event.get('index')}]",
                f"Type: {event.get('mutation_type')} -> {event.get('target')}",
                f"Why: {event.get('rationale')}",
                f"Expected improvement: {event.get('expected_improvement')}",
                f"Risk: {event.get('risk')}",
            ]
        )
    if event_type == "mutation_rejected":
        return "\n".join(
            [
                f"{prefix} / Guardian rejects mutation {event.get('index')}]",
                f"Type: {event.get('mutation_type')} -> {event.get('target')}",
                f"Reason: {event.get('reason')}",
            ]
        )
    if event_type == "mutation_promoted":
        return "\n".join(
            [
                f"{prefix} / Guardian promotes mutation {event.get('index')}]",
                f"Type: {event.get('mutation_type')} -> {event.get('target')}",
                f"Score: {float(event.get('old_score', 0.0)):.4f} -> "
                f"{float(event.get('new_score', 0.0)):.4f}",
                f"Why: {event.get('rationale')}",
            ]
        )
    if event_type == "mutation_rolled_back":
        return "\n".join(
            [
                f"{prefix} / Guardian rolls back mutation {event.get('index')}]",
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
                f"{prefix} / freeze]",
                f"Frozen score: {float(event.get('score', 0.0)):.4f}",
                f"Reason: {event.get('reason')}",
            ]
        )
    return ""


def _format_training_detail_event(event: dict) -> str:
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
                f"Initial hypothesis: {event.get('initial_evaluation_hypothesis')}",
                f"Requirements: {requirements}",
                f"Constraints: {constraints}",
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
        return (
            f"  running {event.get('label')} {event.get('split')} case "
            f"{event.get('case_id')}"
        )
    if event_type == "case_complete":
        return (
            f"  completed {event.get('label')} {event.get('split')} case "
            f"{event.get('case_id')}: {event.get('output_summary')}"
        )
    if event_type == "evaluation_artifacts_written":
        return "\n".join(
            [
                f"Evaluation artifacts written for {event.get('label')}",
                f"  outputs: {event.get('outputs_dir')}",
                f"  traces: {event.get('traces_path')}",
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
                f"Mutation proposed: #{event.get('index')}",
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
                f"Mutation rejected: #{event.get('index')}",
                f"  type: {event.get('mutation_type')} -> {event.get('target')}",
                f"  reviewer: {reviewer}",
                f"  reason: {event.get('reason')}",
            ]
        )
    if event_type == "mutation_promoted":
        hidden = "yes" if event.get("hidden_eval_regression") else "no"
        return "\n".join(
            [
                f"Mutation promoted: #{event.get('index')}",
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
                f"Mutation rolled back: #{event.get('index')}",
                f"  type: {event.get('mutation_type')} -> {event.get('target')}",
                f"  score: {float(event.get('old_score', 0.0)):.4f} -> "
                f"{float(event.get('new_score', 0.0)):.4f}",
                f"  reason: {event.get('reason')}",
            ]
        )
    if event_type == "convergence_decision":
        return "\n".join(
            [
                "Convergence decision",
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
                "Final freeze complete",
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


def _format_transcript_event(trace: dict) -> str:
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


def _infer_scenario_path(frozen_genome: Path) -> Path:
    config_path = frozen_genome.parent / "config_snapshot.yaml"
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if data.get("scenario_path"):
            return Path(data["scenario_path"])
    return Path("scenarios/toy_structured_answer")


if __name__ == "__main__":
    app()


@app.command("audit-signal-leak")
def audit_signal_leak(run_dir: Path, scenario_path: Path) -> None:
    bundle = load_scenario(scenario_path)
    ids = [c.id for c in bundle.final_holdout_cases]
    texts = [str(c.input) for c in bundle.final_holdout_cases]
    ids.extend(c.id for c in bundle.external_benchmark_cases)
    texts.extend(str(c.input) for c in bundle.external_benchmark_cases)
    auditor = SignalLeakAuditor(forbidden_case_ids=ids, forbidden_case_texts=texts)
    for path in nucleus_visible_artifact_paths(run_dir):
        auditor.assert_no_leak_in_file(path)
    typer.echo("Signal leak audit: PASS")
