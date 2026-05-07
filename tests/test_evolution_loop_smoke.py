import json

from stemos.config import Settings
from stemos.evolution.loop import EvolutionLoop
from stemos.genome.loader import load_genome
from stemos.harness.builder import HarnessBuilder
from stemos.harness.runner import HarnessRunner
from stemos.kernel.evaluator import GuardianFitnessEvaluator
from stemos.scenarios.loader import load_scenario
from stemos.scenarios.schema import TaskCase


def test_evolution_loop_smoke_creates_frozen_genome_and_report(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "smoke_001")

    assert result.frozen_genome_path.exists()
    assert result.report_path.exists()
    assert result.final_score > result.baseline_score
    lineage = (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8")
    assert "mutation_promoted" in lineage
    assert "mutation_rejected" in lineage
    assert "freeze" in lineage


def test_demo_final_genome_has_more_than_two_workflow_steps(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "workflow_001")
    frozen = load_genome(result.frozen_genome_path)

    assert len(frozen.workflow) > 2
    assert any(step.id == "review_against_requirements" for step in frozen.workflow)
    assert any(step.id == "revise_final_output" for step in frozen.workflow)


def test_demo_report_contains_organism_shape(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "report_001")
    report = result.report_path.read_text(encoding="utf-8")

    assert "## Organism Shape" in report
    assert "### Initial organism" in report
    assert "### Final organism" in report
    assert "Why This Is Evolution, Not Subagent Orchestration" in report


def test_lineage_contains_promoted_workflow_mutation(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "lineage_workflow_001")
    events = [
        json.loads(line)
        for line in (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert any(
        event["event"] == "mutation_promoted"
        and event["mutation_type"] == "add_workflow_step"
        for event in events
    )


def test_lineage_contains_rejected_or_rolled_back_mutation(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "lineage_reject_001")
    lineage = (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8")

    assert "mutation_rejected" in lineage or "mutation_rolled_back" in lineage


def test_evolution_loop_can_stream_training_events(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )
    events = []

    loop.evolve(
        "scenarios/toy_structured_answer",
        "stream_training_001",
        event_sink=events.append,
    )

    event_types = {event["event"] for event in events}
    assert "harness_trace" in event_types
    assert "mutation_plan" in event_types
    assert "mutation_proposed" in event_types
    assert "mutation_promoted" in event_types
    assert "mutation_rejected" in event_types
    harness_event = next(event for event in events if event["event"] == "harness_trace")
    assert harness_event["case_id"]
    assert harness_event["trace"]["event"] in {
        "environment_materialized",
        "workflow_step",
        "quality_gate",
    }


def test_tiny_task_operator_demo_has_lineage_narrative_and_environment(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/tiny_task_operator", "demo_001")
    frozen = load_genome(result.frozen_genome_path)
    report = result.report_path.read_text(encoding="utf-8")

    assert result.final_score > result.baseline_score
    assert frozen.self_evaluation["enabled"] is True
    assert "artifacts/acceptance_criteria.md" in frozen.environment.required_artifacts
    assert len(frozen.workflow) > 2
    assert frozen.quality_gates
    assert frozen.tools["generated"]
    assert "Differentiation Story" in report
    assert "Guardian promoted" in report
    assert "Guardian rejected" in report
    assert "Baseline genome score" in report
    assert "Evolved genome score" in report


def test_frozen_harness_can_execute_new_input(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )
    result = loop.evolve("scenarios/tiny_task_operator", "execute_001")
    genome = load_genome(result.frozen_genome_path)
    bundle = load_scenario("scenarios/tiny_task_operator")
    harness = HarnessBuilder().materialize(
        genome,
        bundle.scenario,
        workspace_dir=tmp_path / "execute_workspace",
    )

    streamed_traces = []
    run = HarnessRunner().run_case(
        harness,
        TaskCase(
            id="new_input",
            input={"user_request": "Help me plan a focused workday."},
        ),
        trace_sink=streamed_traces.append,
    )

    assert "## Summary" in run.final_output
    assert "## Final Answer" in run.final_output
    workflow_traces = [trace for trace in run.traces if trace["event"] == "workflow_step"]
    assert workflow_traces
    assert workflow_traces[0]["agent_observation"]["goal"]
    assert (
        "hidden chain-of-thought is not recorded"
        in workflow_traces[0]["agent_observation"]["reasoning_boundary"]
    )
    assert streamed_traces == run.traces


def test_frozen_harness_execute_uses_final_output_step(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )
    result = loop.evolve("scenarios/toy_structured_answer", "final_step_001")
    genome = load_genome(result.frozen_genome_path)
    bundle = load_scenario("scenarios/toy_structured_answer")
    harness = HarnessBuilder().materialize(
        genome,
        bundle.scenario,
        workspace_dir=tmp_path / "execute_final_step_workspace",
    )

    run = HarnessRunner().run_case(
        harness,
        TaskCase(
            id="new_input",
            input={"user_request": "Help me plan a focused workday."},
        ),
    )

    assert run.outputs["final_output"] == run.outputs["revise_final_output"]
    assert "## Review Notes Applied" in run.final_output


def test_validation_aware_promotion_and_complexity_pressure():
    bundle = load_scenario("scenarios/tiny_task_operator")
    base_genome = load_genome("src/stemos/genome/default_genome.yaml")
    bloated_genome = base_genome.model_copy(deep=True)
    for index in range(5):
        bloated_genome.environment.required_artifacts.append(f"artifact_{index}.md")

    evaluator = GuardianFitnessEvaluator()

    assert evaluator._complexity_penalty(bloated_genome) > evaluator._complexity_penalty(base_genome)

    harness = HarnessBuilder().materialize(
        base_genome,
        bundle.scenario,
        workspace_dir="workspace/test_validation_aware",
    )
    train_runs = [HarnessRunner().run_case(harness, case) for case in bundle.train_cases]
    validation_runs = [HarnessRunner().run_case(harness, case) for case in bundle.validation_cases]
    result = evaluator.evaluate(base_genome, bundle.scenario, train_runs, validation_runs)

    assert result.validation_score is not None
    assert result.split_policy == "weighted_train_validation_40_60"
