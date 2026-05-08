from __future__ import annotations

import json

from stemos.genome.models import QualityGate, WorkflowStep
from stemos.genome.loader import load_default_genome
from stemos.kernel.convergence import ConvergenceEngine, ConvergencePolicy
from stemos.kernel.guardian import Guardian
from stemos.nucleus.nucleus import build_nucleus_prompt
from stemos.nucleus.schemas import MutationProposal
from stemos.scenarios.loader import load_scenario
from stemos.skills.awm_bridge import AWMBridge
from stemos.skills.extractor import SkillExtractor
from stemos.skills.library import SkillLibrary
from stemos.skills.schema import AtomicSkill


def _atom(
    skill_id: str,
    *,
    name: str = "skill",
    score_lift: float = 0.03,
    domain_tags: list[str] | None = None,
    scenario: str = "scenario_a",
) -> AtomicSkill:
    return AtomicSkill(
        id=skill_id,
        name=name,
        description="Reusable test skill.",
        skill_type="workflow_step",
        genome_fragment={"field_path": "workflow[id=review]", "value": {"id": "review"}},
        domain_tags=domain_tags or ["structured_reasoning"],
        origin_run_id="run_a",
        origin_scenario_name=scenario,
        origin_generation=1,
        score_lift=score_lift,
    )


def test_skill_library_path_threshold_replacement_probation_and_saturation(tmp_path):
    library = SkillLibrary(tmp_path / "atoms.jsonl")

    assert library.add_atom(_atom("too_small", score_lift=0.01)) is False
    assert library.add_atom(_atom("old", name="same", score_lift=0.03)) is True
    assert library.add_atom(_atom("worse", name="same", score_lift=0.02)) is False
    assert library.add_atom(_atom("better", name="same", score_lift=0.07)) is True

    queried = library.query_atoms(
        ["structured_reasoning"],
        current_run_id="run_b",
        current_scenario_name="scenario_b",
        top_k=5,
    )
    assert [atom.id for atom in queried] == ["better"]

    library.open_probation("better", run_id="run_b", scenario_name="scenario_b")
    library.record_observed_lift(
        skill_id="better",
        run_id="run_b",
        observed_score_lift=0.01,
    )
    assert library.tick_probation(run_id="run_b") == []
    assert library.tick_probation(run_id="run_b") == ["better"]
    assert (
        library.query_atoms(
            ["structured_reasoning"],
            current_run_id="run_b",
            current_scenario_name="scenario_b",
            top_k=5,
        )
        == []
    )
    assert library.saturation_score(["no_overlap"]) == 0.0


def test_skill_extractor_uses_stable_field_paths_for_named_genome_parts():
    old = load_default_genome()
    new = old.model_copy(deep=True)
    new.workflow.append(
        WorkflowStep(
            id="review_against_requirements",
            role="Founder",
            action="Review the draft.",
            input_from=["solve_task"],
            output_key="review_notes",
        )
    )
    new.roles[0].instructions = "Solve the task directly, then verify required sections."
    new.quality_gates.append(
        QualityGate(
            name="required_sections_gate",
            description="Check required sections.",
            check_type="schema",
            required=True,
        )
    )
    new.retry_policy["max_attempts"] = 2
    new.self_evaluation["rubric"] = ["final answer"]
    new.memory.setdefault("schema", {})["lesson"] = "string"
    new.tools.setdefault("generated", []).append(
        {
            "name": "requirement_sections_checker",
            "description": "Check requirements.",
            "kind": "generated",
            "path": "checker.py",
            "test_path": "test_checker.py",
        }
    )

    atoms = SkillExtractor().extract(
        old_genome=old,
        new_genome=new,
        mutation_type="test_mutation",
        score_before=0.4,
        score_after=0.5,
        run_id="run_1",
        scenario_name="scenario",
        generation_number=2,
        scenario_domain_tags=["structured"],
    )
    paths = {atom.genome_fragment["field_path"] for atom in atoms}

    assert "workflow[id=review_against_requirements].action" in paths
    assert "roles[name=Founder].instructions" in paths
    assert "quality_gates[name=required_sections_gate].description" in paths
    assert "retry_policy.max_attempts" in paths
    assert "self_evaluation.rubric[0]" in paths
    assert "memory.schema.lesson" in paths
    assert "tools.generated[name=requirement_sections_checker].description" in paths


def test_nucleus_prompt_includes_compact_reusable_skill_section_without_hidden_details():
    skill = _atom("skill_1", score_lift=0.08)
    scenario = load_scenario("scenarios/toy_structured_answer").scenario
    prompt = build_nucleus_prompt(
        scenario_description="structured response",
        current_genome=load_default_genome().model_dump(mode="json"),
        mutation_history=[],
        signal_policy=scenario.signal_policy,
        reusable_skills=[skill],
    )

    assert "Reusable Skills Retrieved From Skill Library" in prompt
    assert "skill | workflow_step | lift=0.080" in prompt
    assert "hidden_eval" not in prompt
    assert "expected_output" not in prompt


def test_convergence_plateau_forces_zero_order_before_freeze_and_logs(tmp_path):
    engine = ConvergenceEngine(
        ConvergencePolicy(
            plateau_window=3,
            plateau_epsilon=0.01,
            max_generations=10,
            absolute_score_threshold=0.99,
            hidden_eval_stability_window=3,
            hidden_eval_stability_epsilon=0.01,
        ),
        run_dir=tmp_path,
        hidden_baseline_score=0.5,
    )

    decisions = []
    for generation in range(3):
        engine.update(
            generation_number=generation,
            validation_score=0.5,
            hidden_eval_score=0.5,
            tokens_used_this_generation=10,
            skill_saturation_score=0.0,
            skill_saturation_available=False,
            zero_order_was_attempted=False,
        )
        decisions.append(engine.evaluate())

    assert decisions[-1].action == "force_zero_order"
    assert decisions[-1].stop is False

    engine.update(
        generation_number=3,
        validation_score=0.5,
        hidden_eval_score=0.5,
        tokens_used_this_generation=10,
        skill_saturation_score=0.0,
        skill_saturation_available=False,
        zero_order_was_attempted=True,
    )
    decision = engine.evaluate()

    assert decision.action == "freeze_best"
    assert decision.stop is True
    assert (tmp_path / "convergence_log.jsonl").read_text(encoding="utf-8").count("\n") == 4


def test_convergence_hard_caps_cannot_be_overridden(tmp_path):
    engine = ConvergenceEngine(
        ConvergencePolicy(max_generations=2, budget_token_limit=1_000_000),
        run_dir=tmp_path / "max",
    )
    for generation in range(2):
        engine.update(
            generation_number=generation,
            validation_score=0.1,
            hidden_eval_score=None,
            tokens_used_this_generation=1,
            skill_saturation_score=1.0,
            zero_order_was_attempted=False,
        )
        decision = engine.evaluate()
    assert decision.action == "stop_max_generations"
    assert decision.stop is True

    budget_engine = ConvergenceEngine(
        ConvergencePolicy(max_generations=20, budget_token_limit=10),
        run_dir=tmp_path / "budget",
    )
    budget_engine.update(
        generation_number=0,
        validation_score=0.99,
        hidden_eval_score=None,
        tokens_used_this_generation=10,
        skill_saturation_score=1.0,
        zero_order_was_attempted=False,
    )
    budget_decision = budget_engine.evaluate()
    assert budget_decision.action == "stop_budget"


def test_awm_bridge_creates_wrapper_and_guardian_evaluates_nested_mutation(tmp_path):
    run_dir = tmp_path / "runs" / "awm_001"
    run_dir.mkdir(parents=True)
    with (run_dir / "intra_adaptation_log.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(
                json.dumps(
                    {
                        "task_id": f"task_{index}",
                        "step_name": "solve_task",
                        "quality_gate_name": "required_sections_gate",
                        "attempt_number": 1,
                        "reflection_text": "Retry with revision for missing requirement sections.",
                        "retry_improved_output": True,
                    }
                )
                + "\n"
            )

    candidates = AWMBridge().analyze("awm_001", {"run_dir": run_dir, "min_occurrences": 3})

    assert len(candidates) == 1
    assert candidates[0].mutation_type == "awm_promoted"
    assert (run_dir / "awm_candidates.jsonl").exists()
    validation = Guardian().validate_mutation(candidates[0], genome=load_default_genome())
    assert validation.allowed
    mutated = Guardian().apply_mutation_safely(load_default_genome(), candidates[0])
    assert mutated.retry_policy["max_attempts"] == 2

    atoms = SkillExtractor().extract(
        old_genome=load_default_genome(),
        new_genome=mutated,
        mutation_type="awm_promoted",
        score_before=0.2,
        score_after=0.25,
        run_id="awm_001",
        scenario_name="scenario",
        generation_number=4,
        scenario_domain_tags=["structured"],
    )
    assert atoms[0].origin_generation == 4
    assert atoms[0].name.startswith("awm_promoted:")
