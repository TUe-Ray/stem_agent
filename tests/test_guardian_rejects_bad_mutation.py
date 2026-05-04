from stemos.genome.loader import load_default_genome
from stemos.kernel.guardian import Guardian
from stemos.nucleus.schemas import MutationProposal
from stemos.scenarios.loader import load_scenario


def test_guardian_rejects_kernel_mutation():
    mutation = MutationProposal(
        mutation_type="edit_tool",
        target="src/stemos/kernel/evaluator.py",
        rationale="Try to change immutable Guardian fitness.",
        expected_improvement="Reward hacking.",
        risk="Unsafe.",
        patch={"path": "src/stemos/kernel/evaluator.py"},
    )

    result = Guardian().validate_mutation(mutation)

    assert result.allowed is False
    assert "protected kernel boundary" in result.reason


def test_guardian_rejects_generated_tool_without_tests():
    mutation = MutationProposal(
        mutation_type="create_tool",
        target="tools.generated",
        rationale="Create a helper tool.",
        expected_improvement="Maybe useful.",
        risk="Untested.",
        patch={
            "name": "helper",
            "description": "No tests.",
            "code": "def run(input_data):\n    return {'ok': True}\n",
        },
    )

    result = Guardian().validate_mutation(mutation)

    assert result.allowed is False
    assert "require tests" in result.reason


def test_guardian_allows_mutable_self_evaluation_mutation():
    mutation = MutationProposal(
        mutation_type="modify_self_evaluation",
        target="self_evaluation",
        rationale="Add self-checks.",
        expected_improvement="Improve requirement coverage.",
        risk="Small cost increase.",
        patch={"enabled": True, "rubric": ["Must include final answer"]},
    )

    result = Guardian().validate_mutation(mutation)

    assert result.allowed is True


def test_guardian_rejects_environment_overgrowth():
    bundle = load_scenario("scenarios/toy_structured_answer")
    genome = load_default_genome()
    artifacts = [f"artifact_{index}.md" for index in range(20)]
    mutation = MutationProposal(
        mutation_type="modify_environment",
        target="environment",
        rationale="Add too many artifacts.",
        expected_improvement="Maybe more state.",
        risk="Overgrowth.",
        patch={"required_artifacts": artifacts},
    )

    result = Guardian().validate_mutation(mutation, genome=genome, scenario=bundle.scenario)

    assert result.allowed is False
    assert "max_environment_artifacts" in result.reason
