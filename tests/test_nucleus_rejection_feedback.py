"""Phase B: Rejection reasons flow back to Nucleus prompt (test_nucleus_rejection_feedback)."""
from stem_agent.kernel.signal_policy import NucleusSignal, SignalPolicy
from stem_agent.nucleus.nucleus import build_nucleus_prompt


def test_rejection_summary_flows_to_nucleus_prompt():
    """When a signal carries last_rejection_summary, it appears in the prompt."""
    sig = NucleusSignal(
        generation=1,
        mutation_type="create_tool",
        direction="unknown",
        aggregate_score=0.5,
        previous_aggregate_score=0.5,
        last_rejection_summary="Generated tool tests failed: test_accepts_covered FAILED",
    )
    prompt = build_nucleus_prompt(
        scenario_description="test",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    assert "last rejection" in prompt.lower(), "prompt must include rejection context"
    assert "test_accepts_covered" in prompt, "rejection detail should be visible"


def test_no_rejection_summary_does_not_inject_garbage():
    """When last_rejection_summary is None, no 'last rejection' line appears."""
    sig = NucleusSignal(
        generation=1,
        mutation_type="add_quality_gate",
        direction="improved",
        aggregate_score=0.6,
        previous_aggregate_score=0.5,
    )
    prompt = build_nucleus_prompt(
        scenario_description="test",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    assert "last rejection" not in prompt.lower(), "no rejection line when summary is None"


def test_long_rejection_is_truncated_in_prompt():
    """Rejection summaries are truncated at ~200 chars in the prompt."""
    long_reason = "x" * 500
    sig = NucleusSignal(
        generation=2,
        mutation_type="create_tool",
        direction="neutral",
        last_rejection_summary=long_reason,
    )
    prompt = build_nucleus_prompt(
        scenario_description="x",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    # Should be truncated (max 200 chars visible)
    assert long_reason[:200] in prompt, "first 200 chars of rejection visible"
    assert long_reason[:400] not in prompt, "rejection truncated beyond 200 chars"
