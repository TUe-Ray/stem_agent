import json

import pytest

from stemos.config import Settings
from stemos.evolution.loop import EvolutionLoop
from stemos.kernel.signal_policy import NucleusSignal, SignalPolicy
from stemos.nucleus.nucleus import build_nucleus_prompt


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
        settings=Settings(offline_mode=True),
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
        settings=Settings(offline_mode=True),
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
