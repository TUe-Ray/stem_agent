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


def test_unsafe_generated_tool_is_rejected():
    mutation = MutationProposal(
        mutation_type="create_tool",
        target="tools.generated",
        rationale="Try unsafe filesystem access.",
        expected_improvement="None.",
        risk="Unsafe import.",
        patch={
            "name": "unsafe_workspace_probe",
            "description": "Should be rejected.",
            "code": "import os\n\ndef run(input_data):\n    return {'cwd': os.getcwd()}\n",
            "test_code": "def test_unsafe_workspace_probe():\n    assert True\n",
        },
    )

    result = Guardian().validate_mutation(mutation)

    assert result.allowed is False
    assert "Forbidden import: os" in result.reason


def test_add_workflow_step_mutation_is_validated_and_applied():
    genome = load_default_genome()
    bundle = load_scenario("scenarios/toy_structured_answer")
    mutation = MutationProposal(
        mutation_type="add_workflow_step",
        target="workflow",
        rationale="The harness currently produces final output without explicit review.",
        expected_improvement="Improve requirement coverage and reduce missing sections.",
        risk="Adds cost and complexity.",
        patch={
            "id": "review_against_requirements",
            "role": "Founder",
            "action": "Review the draft or final output against the scenario requirements and write missing sections or revision notes.",
            "input_from": ["solve_task"],
            "output_key": "review_notes",
        },
    )

    guardian = Guardian()
    result = guardian.validate_mutation(mutation, genome=genome, scenario=bundle.scenario)
    mutated = guardian.apply_mutation_safely(genome, mutation)

    assert result.allowed is True
    assert any(step.id == "review_against_requirements" for step in mutated.workflow)


def test_quality_gate_mutation_is_validated_and_applied():
    genome = load_default_genome()
    mutation = MutationProposal(
        mutation_type="add_quality_gate",
        target="quality_gates",
        rationale="The harness should not finish without checking required output structure.",
        expected_improvement="Reduce missing required sections.",
        risk="May reject valid outputs if too strict.",
        patch={
            "name": "required_sections_gate",
            "description": "Check that the output contains required scenario sections before final delivery.",
            "check_type": "schema",
            "required": True,
        },
    )

    guardian = Guardian()
    result = guardian.validate_mutation(mutation)
    mutated = guardian.apply_mutation_safely(genome, mutation)

    assert result.allowed is True
    assert mutated.quality_gates[0].name == "required_sections_gate"


def test_safe_generated_tool_can_be_activated_after_tests_pass(tmp_path):
    genome = load_default_genome()
    mutation = MutationProposal(
        mutation_type="create_tool",
        target="tools.generated",
        rationale="Add a safe checker tool.",
        expected_improvement="Support evolved quality gates.",
        risk="Small complexity cost.",
        patch={
            "name": "safe_checker",
            "description": "Safe generated checker.",
            "code": (
                "from typing import Any, Dict\n\n"
                "def run(input_data: Dict[str, Any]) -> Dict[str, Any]:\n"
                "    text = str(input_data.get('text', ''))\n"
                "    return {'passed': bool(text), 'length': len(text)}\n"
            ),
            "test_code": (
                "from safe_checker import run\n\n"
                "def test_safe_checker_passes_on_text():\n"
                "    assert run({'text': 'hello'})['passed'] is True\n"
            ),
        },
    )

    guardian = Guardian()
    activation, activated = guardian.activate_generated_tool(
        mutation, tmp_path / "generated_tools"
    )
    mutated = guardian.apply_mutation_safely(genome, activated)

    assert activation.allowed is True
    assert mutated.tools["generated"][0]["name"] == "safe_checker"
    assert mutated.tools["generated"][0]["path"]


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
