from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from stemos.evolution.loop import EvolutionLoop
from stemos.genome.loader import load_genome
from stemos.harness.builder import HarnessBuilder
from stemos.harness.runner import HarnessRunner
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
def evolve(scenario_path: Path, run_id: str = typer.Option(..., "--run-id")) -> None:
    result = EvolutionLoop().evolve(scenario_path, run_id)
    typer.echo(f"Run directory: {result.run_dir}")
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
