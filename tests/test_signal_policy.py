import json

import pytest
import yaml

from stem_agent.config import Settings
from stem_agent.evolution.loop import EvolutionLoop
from stem_agent.kernel.signal_policy import NucleusSignal, SignalPolicy
from stem_agent.nucleus.nucleus import build_nucleus_prompt


def test_signal_policy_blocks_layer2_prompt_content():
    policy = SignalPolicy()

    with pytest.raises(ValueError):
        policy.assert_no_layer2_leak("Here is validation_case_inputs: secret")


def test_nucleus_signal_has_no_layer2_attributes():
    signal = SignalPolicy().build_nucleus_signal(
        generation=1,
        mutation_type="add_workflow_step",
        current_score=0.71,
        previous_score=0.62,
    )

    assert not hasattr(signal, "validation_case_inputs")
    assert not hasattr(signal, "validation_case_expected_outputs")
    assert not hasattr(signal, "guardian_criterion_scores")
    assert not hasattr(signal, "hidden_eval_scores")
    assert not hasattr(signal, "per_case_scores")


def test_build_nucleus_prompt_uses_signal_history_without_layer2_content():
    policy = SignalPolicy()
    signal = NucleusSignal(
        generation=1,
        mutation_type="modify_self_evaluation",
        direction="improved",
        aggregate_score=0.71,
        previous_aggregate_score=0.62,
        score_delta=0.09,
    )

    prompt = build_nucleus_prompt(
        scenario_description="structured_reasoning: solve math problems",
        current_genome={
            "genome_version": 1,
            "name": "demo",
            "roles": [{"name": "solver", "description": "Solves tasks"}],
            "workflow": [{"id": "solve", "role": "solver"}],
            "quality_gates": [],
            "tools": {},
            "environment": {"required_artifacts": []},
        },
        mutation_history=[signal],
        signal_policy=policy,
    )

    policy.assert_no_layer2_leak(prompt)
    assert "expected_output" not in prompt
    assert "Generation 1: modify_self_evaluation -> improved" in prompt


def test_signal_policy_override_removes_aggregate_scores_from_lineage(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve(
        "scenarios/toy_structured_answer",
        "blind_signal",
        signal_policy_override={"layer_1_enabled": False},
    )
    events = [
        json.loads(line)
        for line in (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    signals = [
        event["nucleus_signal"]
        for event in events
        if event.get("event") == "mutation_promoted" and event.get("nucleus_signal")
    ]

    assert signals
    assert all(signal["aggregate_score"] is None for signal in signals)
    assert all(signal["score_delta"] is None for signal in signals)


def test_default_signal_policy_exposes_aggregate_scores_for_promotions(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "signal_guided")
    events = [
        json.loads(line)
        for line in (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    signals = [
        event["nucleus_signal"]
        for event in events
        if event.get("event") == "mutation_promoted" and event.get("nucleus_signal")
    ]

    assert signals
    assert all(isinstance(signal["aggregate_score"], float) for signal in signals)


def test_hidden_evaluation_details_do_not_enter_nucleus_visible_artifacts(tmp_path):
    scenario_dir = tmp_path / "hidden_signal_scenario"
    scenario_dir.mkdir()
    scenario = {
        "scenario": {
            "name": "hidden_signal_scenario",
            "description": "Exercise hidden evaluation leakage boundaries.",
            "task_class": "structured_answer",
        },
        "input_format": {
            "type": "text",
            "fields": [{"name": "user_request", "required": True}],
        },
        "expected_output": {
            "type": "markdown",
            "requirements": ["Must include a short summary", "Must include final answer"],
        },
        "constraints": ["Prefer actionable output"],
        "available_builtin_tools": ["call_model"],
        "success_criteria": [
            {
                "name": "requirement_coverage",
                "weight": 1.0,
                "description": "Output satisfies required sections.",
            }
        ],
        "evolution": {
            "max_generations": 1,
            "patience": 1,
            "min_delta": 0.03,
            "max_mutations_per_generation": 1,
            "max_cost_usd": 2.0,
        },
    }
    (scenario_dir / "scenario.yaml").write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )
    (scenario_dir / "train_cases.jsonl").write_text(
        json.dumps(
            {
                "id": "train_001",
                "input": {"user_request": "Prepare a short plan."},
                "reference_notes": "Normal train notes.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "validation_cases.jsonl").write_text(
        json.dumps(
            {
                "id": "val_001",
                "input": {"user_request": "Prepare another short plan."},
                "reference_notes": "Normal validation notes.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hidden_secret = "SECRET_HIDDEN_EXPECTED_OUTPUT"
    (scenario_dir / "hidden_cases.jsonl").write_text(
        json.dumps(
            {
                "id": "hidden_001",
                "input": {"user_request": "Prepare a hidden short plan."},
                "expected_output": hidden_secret,
                "reference_notes": hidden_secret,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve(scenario_dir, "hidden_signal")
    lineage = (result.run_dir / "lineage.jsonl").read_text(encoding="utf-8")
    llm_calls = (result.run_dir / "llm_calls.jsonl").read_text(encoding="utf-8")

    assert (result.run_dir / "hidden_eval_log.jsonl").exists()
    assert hidden_secret not in lineage
    assert hidden_secret not in llm_calls
