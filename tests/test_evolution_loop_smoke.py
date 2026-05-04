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

    run = HarnessRunner().run_case(
        harness,
        TaskCase(
            id="new_input",
            input={"user_request": "Help me plan a focused workday."},
        ),
    )

    assert "## Summary" in run.final_output
    assert "## Final Answer" in run.final_output


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
    assert result.split_policy == "weighted_train_validation_35_65"
