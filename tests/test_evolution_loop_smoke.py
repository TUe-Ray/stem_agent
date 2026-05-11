import json

from stem_agent.config import Settings
from stem_agent.evolution.loop import EvolutionLoop
from stem_agent.genome.loader import load_genome
from stem_agent.genome.models import QualityGate, WorkflowStep
from stem_agent.harness.builder import HarnessBuilder
from stem_agent.harness.runner import HarnessRunResult, HarnessRunner
from stem_agent.harness.role_runner import RoleRunner
from stem_agent.kernel.evaluator import GuardianFitnessEvaluator
from stem_agent.nucleus.model_client import ModelClient
from stem_agent.nucleus.schemas import MutationPlan, MutationProposal
from stem_agent.scenarios.loader import load_scenario
from stem_agent.scenarios.schema import SuccessCriterion, TaskCase


def test_evolution_loop_smoke_creates_frozen_genome_and_report(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "smoke_001")

    assert result.frozen_genome_path.exists()
    assert result.report_path.exists()
    assert result.final_score > result.baseline_score
    lineage = (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8")
    assert "mutation_promoted" in lineage
    assert "mutation_rejected" in lineage or "mutation_rolled_back" in lineage
    assert "freeze" in lineage


def test_demo_final_genome_has_more_than_two_workflow_steps(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "workflow_001")
    frozen = load_genome(result.frozen_genome_path)

    assert len(frozen.workflow) > 2
    assert any(step.id == "review_against_requirements" for step in frozen.workflow)
    assert any(step.id == "revise_final_output" for step in frozen.workflow)


def test_demo_report_contains_organism_shape(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
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
        settings=Settings(test_mode=True),
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
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "lineage_reject_001")
    lineage = (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8")

    assert "mutation_rejected" in lineage or "mutation_rolled_back" in lineage


def test_evolution_loop_rejects_duplicate_workflow_step_without_crashing(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    def duplicate_step_plan(**kwargs):
        _ = kwargs
        return MutationPlan(
            summary="Live-like bad mutation reused an existing workflow id.",
            proposed_mutations=[
                MutationProposal(
                    mutation_type="add_workflow_step",
                    target="workflow",
                    rationale="Reuse solve_task to simulate malformed live model output.",
                    expected_improvement="",
                    risk="Duplicate workflow id.",
                    patch={
                        "id": "solve_task",
                        "role": "Founder",
                        "action": "Duplicate the existing solve step.",
                        "input_from": ["understand_task"],
                        "output_key": "duplicate_output",
                    },
                )
            ],
        )

    loop.nucleus.plan = duplicate_step_plan

    result = loop.evolve("scenarios/toy_structured_answer", "duplicate_step_001")
    events = [
        json.loads(line)
        for line in (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result.status == "FROZEN"
    assert any(
        event["event"] == "mutation_rejected"
        and "duplicate workflow step id solve_task" in event["reason"]
        for event in events
    )


def test_evolution_loop_can_stream_training_events(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
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
    assert {"mutation_rejected", "mutation_rolled_back"} & event_types
    harness_event = next(event for event in events if event["event"] == "harness_trace")
    assert harness_event["case_id"]
    assert harness_event["trace"]["event"] in {
        "environment_materialized",
        "workflow_step",
        "quality_gate",
    }


def test_tiny_task_operator_demo_has_lineage_narrative_and_environment(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/tiny_task_operator", "demo_001")
    frozen = load_genome(result.frozen_genome_path)
    report = result.report_path.read_text(encoding="utf-8")
    lineage = (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8")

    assert result.final_score > result.baseline_score
    assert frozen.self_evaluation["enabled"] is True
    assert "artifacts/acceptance_criteria.md" in frozen.environment.required_artifacts
    assert len(frozen.workflow) > 2
    assert frozen.quality_gates
    assert frozen.tools["generated"] or '"mutation_type": "create_tool"' in lineage
    assert "Differentiation Story" in report
    assert "Guardian promoted" in report
    assert "Rejected Or Rolled Back Mutations" in report
    assert "Baseline genome score" in report
    assert "Evolved genome score" in report


def test_frozen_harness_can_execute_new_input(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
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
    run = HarnessRunner(RoleRunner(ModelClient(test_mode=True))).run_case(
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
        settings=Settings(test_mode=True),
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

    run = HarnessRunner(RoleRunner(ModelClient(test_mode=True))).run_case(
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
    base_genome = load_genome("src/stem_agent/genome/default_genome.yaml")
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
    runner = HarnessRunner(RoleRunner(ModelClient(test_mode=True)))
    train_runs = [runner.run_case(harness, case) for case in bundle.train_cases]
    validation_runs = [runner.run_case(harness, case) for case in bundle.validation_cases]
    result = evaluator.evaluate(base_genome, bundle.scenario, train_runs, validation_runs)

    assert result.validation_score is not None
    assert result.split_policy == "weighted_train_validation_40_60"


def test_generation_patience_decrements_once_for_no_best_improvement(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    assert (
        loop._finish_generation_patience(
            3,
            generation_improved_best=False,
        )
        == 2
    )
    assert (
        loop._finish_generation_patience(
            3,
            generation_improved_best=True,
        )
        == 3
    )


def test_requirement_matching_is_shared_by_evaluator_and_quality_gate(tmp_path):
    bundle = load_scenario("scenarios/toy_structured_answer")
    genome = load_genome("src/stem_agent/genome/default_genome.yaml")
    harness = HarnessBuilder().materialize(
        genome,
        bundle.scenario,
        workspace_dir=tmp_path / "shared_requirement_workspace",
    )
    evaluator = GuardianFitnessEvaluator()
    runner = HarnessRunner(RoleRunner(ModelClient(test_mode=True)))

    output = "## Summary\nUseful overview.\n\n## Final Answer\nA concrete recommendation."

    assert evaluator._requirement_satisfied("Must include a short summary", output)
    assert evaluator._requirement_satisfied("Must include final answer", output)
    missing = runner._missing_required_sections(harness, output.lower())
    assert "Must include a short summary" not in missing
    assert "Must include final answer" not in missing


def test_self_evaluation_produces_trace_and_metric(tmp_path):
    bundle = load_scenario("scenarios/toy_structured_answer")
    genome = load_genome("src/stem_agent/genome/default_genome.yaml")
    genome.self_evaluation = {
        "enabled": True,
        "rubric": bundle.scenario.expected_output.requirements,
    }
    harness = HarnessBuilder().materialize(
        genome,
        bundle.scenario,
        workspace_dir=tmp_path / "self_eval_workspace",
    )
    runner = HarnessRunner(RoleRunner(ModelClient(test_mode=True)))
    run = runner.run_case(harness, bundle.train_cases[0])

    assert any(trace.get("event") == "self_evaluation" for trace in run.traces)
    result = GuardianFitnessEvaluator().evaluate(
        genome,
        bundle.scenario,
        [run],
        audit=False,
    )
    assert result.metrics["self_review_usage"] > 0.0
    assert result.metrics["safeguard_effectiveness"] > 0.3


def test_reference_notes_alignment_rewards_case_specific_guidance():
    bundle = load_scenario("scenarios/toy_structured_answer")
    genome = load_genome("src/stem_agent/genome/default_genome.yaml")
    evaluator = GuardianFitnessEvaluator()
    generic = HarnessRunResult(
        case_id="reference_alignment",
        case_input={"user_request": "Plan a morning routine."},
        reference_notes="Should include timed steps, tradeoffs, and final recommendation.",
        final_output="## Summary\nPlan the morning.\n\n## Steps\n1. Do the work.\n\n## Final Answer\nProceed.",
    )
    specific = generic.model_copy(
        update={
            "final_output": (
                "## Summary\nPlan a morning routine with timed steps.\n\n"
                "## Steps\n1. Make a timed checklist.\n2. Name tradeoffs.\n\n"
                "## Final Answer\nUse the final recommendation after protecting a buffer."
            )
        }
    )

    generic_score = evaluator.evaluate(genome, bundle.scenario, [generic], audit=False)
    specific_score = evaluator.evaluate(genome, bundle.scenario, [specific], audit=False)

    assert specific_score.metrics["reference_alignment"] > generic_score.metrics["reference_alignment"]
    assert specific_score.promotion_score > generic_score.promotion_score


def test_success_criteria_affect_default_evaluator_score():
    bundle = load_scenario("scenarios/toy_structured_answer")
    genome = load_genome("src/stem_agent/genome/default_genome.yaml")
    run = TaskCase(
        id="criterion_case",
        input={"user_request": "Plan a budget meeting about a vendor renewal."},
    )
    harness_run = HarnessRunResult(
        case_id=run.id,
        case_input=dict(run.input),
        final_output=(
            "## Summary\nGeneric response.\n\n"
            "## Concrete Steps\n- Do the work.\n\n"
            "## Final Answer\nProceed."
        ),
    )
    evaluator = GuardianFitnessEvaluator()

    requirement_scenario = bundle.scenario.model_copy(
        update={
            "success_criteria": [
                SuccessCriterion(
                    name="requirement_coverage",
                    weight=1.0,
                    description="Only required sections matter.",
                )
            ]
        },
        deep=True,
    )
    usefulness_scenario = bundle.scenario.model_copy(
        update={
            "success_criteria": [
                SuccessCriterion(
                    name="usefulness",
                    weight=1.0,
                    description="Specific, actionable usefulness matters.",
                )
            ]
        },
        deep=True,
    )

    requirement_score = evaluator.evaluate(
        genome,
        requirement_scenario,
        [harness_run],
        [],
        audit=False,
    ).promotion_score
    usefulness_score = evaluator.evaluate(
        genome,
        usefulness_scenario,
        [harness_run],
        [],
        audit=False,
    ).promotion_score

    assert requirement_score > usefulness_score


def test_stem_process_metrics_reward_specialized_harness_shape():
    bundle = load_scenario("scenarios/toy_structured_answer")
    base_genome = load_genome("src/stem_agent/genome/default_genome.yaml")
    specialized = base_genome.model_copy(deep=True)
    specialized.task_diagnosis = {
        "task_type": "structured answer planning",
        "expected_task_solving_pattern": "diagnose request, draft concrete plan, review constraints, revise final answer",
        "likely_failure_modes": ["generic advice", "missed constraint", "weak final answer"],
        "likely_needed_capabilities": ["input-specific planning", "self-review", "quality gate"],
        "initial_architecture_hypothesis": "Founder workflow with review and final revision step",
        "initial_evaluation_hypothesis": "Reward actionability, input specificity, requirement coverage, and safeguard usage",
    }
    specialized.workflow.append(
        WorkflowStep(
            id="review_against_requirements",
            role="Founder",
            action="Review the draft against requirements, constraints, and likely failure modes.",
            input_from=["final_output"],
            output_key="review_notes",
        )
    )
    specialized.workflow.append(
        WorkflowStep(
            id="revise_final_output",
            role="Founder",
            action="Revise the response into a specific final answer with concrete steps.",
            input_from=["final_output", "review_notes"],
            output_key="final_output",
        )
    )
    specialized.self_evaluation = {
        "enabled": True,
        "rubric": ["specific", "actionable", "constraint-aware"],
    }
    specialized.quality_gates.append(
        QualityGate(
            name="final_answer_quality",
            description="Checks that the final answer is specific, actionable, and constraint-aware.",
            check_type="manual_placeholder",
        )
    )
    specialized.environment.required_artifacts.append("artifacts/decision_log.md")

    output = (
        "## Summary\nPlan a focused 20-minute morning routine before school drop-off.\n\n"
        "## Concrete Steps\n"
        "1. Put bags and shoes by the door first.\n"
        "2. Prepare a simple breakfast and skip optional chores.\n"
        "3. Keep five minutes as a fallback buffer.\n\n"
        "## Review Notes Applied\nChecked time, parent context, and stress constraint.\n\n"
        "## Final Answer\nUse a short checklist, protect the buffer, and review tomorrow."
    )
    generic_run = HarnessRunResult(
        case_id="stem_metric_case",
        case_input={"user_request": "Plan a less stressful 20-minute school morning routine."},
        final_output=output,
        outputs={"final_output": output},
        traces=[
            {"event": "workflow_step", "step_id": "understand_task"},
            {"event": "workflow_step", "step_id": "solve_task"},
        ],
    )
    specialized_run = generic_run.model_copy(
        deep=True,
        update={
            "traces": [
                {"event": "environment_materialized", "artifacts": ["artifacts/decision_log.md"]},
                {"event": "workflow_step", "step_id": "understand_task"},
                {"event": "workflow_step", "step_id": "solve_task"},
                {"event": "workflow_step", "step_id": "review_against_requirements"},
                {"event": "quality_gate", "passed": True},
                {"event": "workflow_step", "step_id": "revise_final_output"},
            ]
        },
    )
    evaluator = GuardianFitnessEvaluator()

    generic = evaluator.evaluate(base_genome, bundle.scenario, [generic_run], audit=False)
    evolved = evaluator.evaluate(specialized, bundle.scenario, [specialized_run], audit=False)

    assert evolved.metrics["stem_process_quality"] > generic.metrics["stem_process_quality"]
    assert evolved.metrics["diagnosis_quality"] > generic.metrics["diagnosis_quality"]
    assert evolved.metrics["safeguard_effectiveness"] > generic.metrics["safeguard_effectiveness"]
    assert evolved.promotion_score > generic.promotion_score


def test_exact_answer_criteria_remain_primary_over_stem_process_bonus():
    bundle = load_scenario("scenarios/gsm8k_demo")
    base_genome = load_genome("src/stem_agent/genome/default_genome.yaml")
    specialized = base_genome.model_copy(deep=True)
    specialized.task_diagnosis = {
        "task_type": "structured math reasoning",
        "expected_task_solving_pattern": "extract quantities, compute carefully, verify final numeric answer",
        "likely_failure_modes": ["arithmetic error", "unsupported number"],
        "likely_needed_capabilities": ["calculation", "verification"],
        "initial_architecture_hypothesis": "Add verification before final answer",
        "initial_evaluation_hypothesis": "Exact final answer is the primary score",
    }
    specialized.self_evaluation = {"enabled": True, "rubric": ["exact answer"]}
    specialized.quality_gates.append(
        QualityGate(
            name="answer_check",
            description="Checks arithmetic and final numeric answer.",
            check_type="manual_placeholder",
        )
    )

    correct_run = HarnessRunResult(
        case_id="gsm_correct",
        case_input={"problem": "Ray has 40 apples and buys 2 more. How many apples?"},
        expected_output="42",
        final_output="40 + 2 = 42\n\nFinal answer: 42",
        traces=[{"event": "workflow_step", "step_id": "solve_task"}],
    )
    wrong_run = HarnessRunResult(
        case_id="gsm_wrong",
        case_input={"problem": "Ray has 40 apples and buys 2 more. How many apples?"},
        expected_output="42",
        final_output="40 + 2 = 41\n\nFinal answer: 41",
        traces=[
            {"event": "workflow_step", "step_id": "solve_task"},
            {"event": "quality_gate", "passed": True},
        ],
    )
    evaluator = GuardianFitnessEvaluator()

    correct = evaluator.evaluate(base_genome, bundle.scenario, [correct_run], audit=False)
    wrong = evaluator.evaluate(specialized, bundle.scenario, [wrong_run], audit=False)

    assert correct.metrics["task_performance"] > wrong.metrics["task_performance"]
    assert correct.promotion_score > wrong.promotion_score
