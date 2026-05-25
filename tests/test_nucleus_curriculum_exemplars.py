"""Phase C: Nucleus curriculum exemplars in prompt (test_nucleus_curriculum_exemplars)."""
from stem_agent.kernel.signal_policy import NucleusSignal, SignalPolicy
from stem_agent.nucleus.nucleus import build_nucleus_prompt


def test_create_tool_exemplar_in_prompt():
    """The Nucleus prompt contains a working create_tool exemplar."""
    sig = NucleusSignal(
        generation=0,
        mutation_type="baseline",
        direction="unknown",
    )
    prompt = build_nucleus_prompt(
        scenario_description="structured QA task",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
        failure_patterns=[],
    )
    assert "create_tool" in prompt, "prompt should mention create_tool as a mutation type"
    assert "requirement_sections_checker" in prompt, "working exemplar name should be visible"
    assert "from typing import Any, Dict" in prompt, "exemplar code should be copyable"


def test_mutation_order_hint_in_prompt():
    """The prompt suggests a curriculum order for structured-answer tasks."""
    sig = NucleusSignal(
        generation=0,
        mutation_type="baseline",
        direction="unknown",
    )
    prompt = build_nucleus_prompt(
        scenario_description="structured QA task",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    assert "self-evaluation" in prompt.lower(), "self-evaluation should be first in curriculum"
    assert "review step" in prompt.lower(), "review step should be in curriculum"
    assert "quality gate" in prompt.lower(), "quality gate should be in curriculum"


def test_exemplar_has_all_required_patch_fields():
    """The create_tool exemplar includes name, description, code, test_code."""
    sig = NucleusSignal(
        generation=0,
        mutation_type="baseline",
        direction="unknown",
    )
    prompt = build_nucleus_prompt(
        scenario_description="test",
        current_genome={"name": "g", "roles": [], "workflow": [], "tools": {}},
        mutation_history=[sig],
        signal_policy=SignalPolicy(),
    )
    # All four key fields should appear in the exemplar block
    assert "name:" in prompt, "name field in exemplar"
    assert "description:" in prompt, "description field in exemplar"
    assert "code: |" in prompt, "code field in exemplar (YAML literal)"
    assert "test_code: |" in prompt, "test_code field in exemplar (YAML literal)"
