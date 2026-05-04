from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from stemos.evolution.control import EvolutionControl
from stemos.evolution.loop import EvolutionLoop
from stemos.genome.loader import load_genome
from stemos.harness.builder import HarnessBuilder
from stemos.harness.runner import HarnessRunner
from stemos.kernel.versioning import GitProvenance, GitRunConfig
from stemos.scenarios.loader import load_scenario
from stemos.scenarios.schema import TaskCase

app = typer.Typer(help="StemOS command line interface.")


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


@app.command()
def evolve(
    scenario_path: Path,
    run_id: str = typer.Option(..., "--run-id"),
    git_branch: bool = typer.Option(False, "--git-branch"),
    git_commit: bool = typer.Option(False, "--git-commit"),
    git_push: bool = typer.Option(False, "--git-push"),
) -> None:
    result = EvolutionLoop().evolve(
        scenario_path,
        run_id,
        git_branch=git_branch,
        git_commit=git_commit,
        git_push=git_push,
    )
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
        typer.echo(report_path.read_text(encoding="utf-8"))
        return
    if not lineage_path.exists():
        raise typer.BadParameter(f"Missing run artifacts at {run_path}")
    typer.echo(lineage_path.read_text(encoding="utf-8"))


@app.command()
def compare(run_path: Path) -> None:
    baseline = _read_eval(run_path / "generation_000" / "eval_result.json")
    final = _read_eval(run_path / "final_evaluation" / "eval_result.json")
    typer.echo(f"Baseline score: {baseline['promotion_score']:.4f}")
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
def execute(
    frozen_genome: Path,
    input: str = typer.Option(..., "--input"),
    scenario_path: Optional[Path] = typer.Option(None, "--scenario-path"),
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
    result = HarnessRunner().run_case(harness, case)
    typer.echo(result.final_output)


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


def _infer_scenario_path(frozen_genome: Path) -> Path:
    config_path = frozen_genome.parent / "config_snapshot.yaml"
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if data.get("scenario_path"):
            return Path(data["scenario_path"])
    return Path("scenarios/toy_structured_answer")


if __name__ == "__main__":
    app()
