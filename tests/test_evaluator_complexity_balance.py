"""Phase D: Complexity penalty — first tool/gate/role is free (test_evaluator_complexity_balance)."""
from stem_agent.genome.models import Genome
from stem_agent.kernel.evaluator import GuardianFitnessEvaluator


def make_base_genome(**overrides) -> Genome:
    """Build a minimal valid genome, override specific fields."""
    base = {
        "genome_version": 0,
        "name": "test",
        "scenario_name": "test",
        "task_diagnosis": {},
        "roles": [],
        "workflow": [],
        "tools": {"builtin": [], "generated": []},
        "memory": {},
        "quality_gates": [],
        "self_evaluation": {"rubric": ""},
        "retry_policy": {"max_attempts": 1, "revise_on_failure": False},
        "stop_rule": {},
        "environment": {
            "workspace_layout": [],
            "required_artifacts": [],
            "artifact_purpose": {},
            "file_templates": {},
            "cleanup_policy": "keep_run_artifacts",
        },
        **overrides,
    }
    return Genome.model_validate(base)


class TestFirstToolFree:
    def test_zero_tools_zero_penalty(self):
        """No generated tools → no penalty."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        genome = make_base_genome()
        penalty = e._complexity_penalty(genome)
        assert penalty == 0.0, f"Expected 0 penalty, got {penalty}"

    def test_first_generated_tool_is_free(self):
        """First generated tool has zero penalty (max(len(tools)-1, 0) = 0)."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        genome = make_base_genome(
            tools={"builtin": [], "generated": [{"name": "checker", "code": "pass", "test_code": "pass", "description": "test"}]}
        )
        penalty = e._complexity_penalty(genome)
        assert penalty == 0.0, f"First tool should be free, got {penalty}"


class TestFirstGateFree:
    def test_first_quality_gate_is_free(self):
        """First quality gate has zero penalty (max(len(gates)-1, 0) = 0)."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        genome = make_base_genome(
            quality_gates=[{"name": "check_output", "description": "check", "check_type": "schema"}]
        )
        penalty = e._complexity_penalty(genome)
        assert penalty == 0.0, f"First gate should be free, got {penalty}"


class TestComplexityGrows:
    def test_second_tool_costs(self):
        """Second generated tool triggers penalty."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        genome = make_base_genome(
            tools={"builtin": [], "generated": [
                {"name": "t1", "code": "pass", "test_code": "pass", "description": "a"},
                {"name": "t2", "code": "pass", "test_code": "pass", "description": "b"},
            ]}
        )
        penalty = e._complexity_penalty(genome)
        assert penalty > 0.0, f"Second tool should cost something, got {penalty}"

    def test_second_gate_costs(self):
        """Second quality gate triggers penalty."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        genome = make_base_genome(
            quality_gates=[
                {"name": "g1", "description": "a", "check_type": "schema"},
                {"name": "g2", "description": "b", "check_type": "python"},
            ]
        )
        penalty = e._complexity_penalty(genome)
        assert penalty > 0.0, f"Second gate should cost something, got {penalty}"
