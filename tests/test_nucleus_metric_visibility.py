"""Phase B: Nucleus prompt includes aggregate metric breakdown (test_nucleus_metric_visibility)."""
from stem_agent.kernel.signal_policy import NucleusSignal, SignalPolicy
from stem_agent.nucleus.nucleus import build_nucleus_prompt


def test_prompt_exposes_aggregate_metric_breakdown():
    """NucleusSignal with metric_breakdown + weakest_metrics renders into the prompt."""
    sig = NucleusSignal(
        generation=1,
        mutation_type="evaluation",
        direction="improved",
        aggregate_score=0.42,
        previous_aggregate_score=0.40,
        metric_breakdown={
            "requirement_coverage": 0.35,
            "actionability": 0.70,
            "workflow_completion": 0.20,
        },
        weakest_metrics=["workflow_completion", "requirement_coverage"],
    )
    prompt = build_nucleus_prompt(
        scenario_description="test scenario",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    assert "workflow_completion" in prompt, "prompt must mention weakest metric"
    assert "requirement_coverage" in prompt, "prompt must mention weakest metric"
    assert "0.20" in prompt or "0.2" in prompt, "weakest metric value should be visible"
    assert "weakest" in prompt.lower(), "prompt should indicate weakest metrics"
    assert "low metrics" in prompt.lower(), "prompt should flag low metrics"


def test_prompt_without_metric_breakdown_still_works():
    """NucleusSignal without metric_breakdown does not crash."""
    sig = NucleusSignal(
        generation=1,
        mutation_type="evaluation",
        direction="neutral",
    )
    prompt = build_nucleus_prompt(
        scenario_description="test",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    assert "Generation 1" in prompt, "basic history should render"
    assert "test" in prompt, "scenario should appear"


def test_prompt_handles_empty_metrics_gracefully():
    """Empty metric_breakdown and weakest_metrics don't break the prompt."""
    sig = NucleusSignal(
        generation=0,
        mutation_type="baseline",
        direction="unknown",
        metric_breakdown={},
        weakest_metrics=[],
    )
    prompt = build_nucleus_prompt(
        scenario_description="x",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    # Should not crash — empty metrics just produce no extra lines
    assert "Generation 0" in prompt


def test_policy_disabled_hides_score_details():
    """When layer_1 is disabled, score details are hidden."""
    sig = NucleusSignal(
        generation=2,
        mutation_type="add_quality_gate",
        direction="improved",
        aggregate_score=0.60,
        previous_aggregate_score=0.55,
    )
    policy = SignalPolicy(layer_1_enabled=False)
    prompt = build_nucleus_prompt(
        scenario_description="x",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=policy,
    )
    # Direction still shown, but numeric scores are not
    assert "add_quality_gate" in prompt
    assert "improved" in prompt
    assert "0.60" not in prompt  # scores hidden when layer_1 disabled
