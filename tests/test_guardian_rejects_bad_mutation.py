from stem_agent.genome.loader import load_default_genome
from stem_agent.genome.models import QualityGate, RoleSpec
from stem_agent.kernel.guardian import Guardian
from stem_agent.nucleus.schemas import MutationProposal
from stem_agent.scenarios.loader import load_scenario


def test_guardian_rejects_kernel_mutation():
    mutation = MutationProposal(
        mutation_type="edit_tool",
        target="src/stem_agent/kernel/evaluator.py",
        rationale="Try to change immutable Guardian fitness.",
        expected_improvement="Reward hacking.",
        risk="Unsafe.",
        patch={"path": "src/stem_agent/kernel/evaluator.py"},
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


def test_guardian_rejects_malformed_quality_gate_mutation_before_apply():
    mutation = MutationProposal(
        mutation_type="add_quality_gate",
        target="",
        rationale="Add a quality gate, but the LLM returned an empty patch.",
        expected_improvement="",
        risk="",
        patch={},
    )

    result = Guardian().validate_mutation(mutation)

    assert result.allowed is False
    assert "Invalid add_quality_gate patch" in result.reason
    assert "name" in result.reason
    assert "description" in result.reason
    assert "check_type" in result.reason


def test_guardian_rejects_edit_mutation_without_existing_target():
    genome = load_default_genome()
    mutation = MutationProposal(
        mutation_type="edit_workflow_step",
        target="missing_step",
        rationale="Edit a nonexistent workflow step.",
        expected_improvement="",
        risk="",
        patch={"action": "Rewrite the missing step."},
    )

    result = Guardian().validate_mutation(mutation, genome=genome)

    assert result.allowed is False
    assert "missing target missing_step" in result.reason


def test_guardian_rejects_duplicate_workflow_step_id_before_apply():
    genome = load_default_genome()
    mutation = MutationProposal(
        mutation_type="add_workflow_step",
        target="workflow",
        rationale="Live model reused an existing step id.",
        expected_improvement="",
        risk="",
        patch={
            "id": "solve_task",
            "role": "Founder",
            "action": "Duplicate the solve step.",
            "input_from": ["understand_task"],
            "output_key": "duplicate_output",
        },
    )

    result = Guardian().validate_mutation(mutation, genome=genome)

    assert result.allowed is False
    assert "duplicate workflow step id solve_task" in result.reason


def test_guardian_rejects_edit_workflow_step_to_existing_id():
    genome = load_default_genome()
    mutation = MutationProposal(
        mutation_type="edit_workflow_step",
        target="understand_task",
        rationale="Live model tried to rename a step to an existing id.",
        expected_improvement="",
        risk="",
        patch={"id": "solve_task"},
    )

    result = Guardian().validate_mutation(mutation, genome=genome)

    assert result.allowed is False
    assert "duplicate workflow step id solve_task" in result.reason


def test_guardian_rejects_duplicate_role_quality_gate_and_tool_names():
    genome = load_default_genome()
    genome.roles.append(
        RoleSpec(
            name="Reviewer",
            description="Reviews drafts.",
            instructions="Review drafts.",
            allowed_tools=[],
        )
    )
    genome.quality_gates.append(
        QualityGate(
            name="required_sections_gate",
            description="Check required sections.",
            check_type="schema",
            required=True,
        )
    )
    genome.tools.setdefault("generated", []).append(
        {
            "name": "requirement_checker",
            "description": "Existing generated checker.",
            "kind": "generated",
            "path": "checker.py",
            "test_path": "test_checker.py",
        }
    )
    guardian = Guardian()

    duplicate_role = guardian.validate_mutation(
        MutationProposal(
            mutation_type="edit_role",
            target="Reviewer",
            patch={"name": "Founder"},
        ),
        genome=genome,
    )
    duplicate_gate = guardian.validate_mutation(
        MutationProposal(
            mutation_type="add_quality_gate",
            target="quality_gates",
            patch={
                "name": "required_sections_gate",
                "description": "Duplicate gate.",
                "check_type": "schema",
                "required": True,
            },
        ),
        genome=genome,
    )
    duplicate_tool = guardian.validate_mutation(
        MutationProposal(
            mutation_type="create_tool",
            target="tools.generated",
            patch={
                "name": "requirement_checker",
                "description": "Duplicate generated checker.",
                "code": "def run(input_data):\n    return {'ok': True}\n",
                "test_code": "from requirement_checker import run\n\ndef test_run():\n    assert run({})['ok']\n",
            },
        ),
        genome=genome,
    )

    assert duplicate_role.allowed is False
    assert "duplicate role name Founder" in duplicate_role.reason
    assert duplicate_gate.allowed is False
    assert "duplicate quality gate name required_sections_gate" in duplicate_gate.reason
    assert duplicate_tool.allowed is False
    assert "duplicate generated tool name requirement_checker" in duplicate_tool.reason


def test_guardian_rejects_replace_genome_with_duplicate_workflow_ids_before_apply():
    genome = load_default_genome()
    replacement = genome.model_dump(mode="json")
    replacement["workflow"].append(dict(replacement["workflow"][0]))
    mutation = MutationProposal(
        mutation_type="replace_genome",
        target="genome",
        rationale="Whole-genome proposal duplicated a workflow id.",
        expected_improvement="",
        risk="",
        patch={"genome": replacement},
    )

    result = Guardian().validate_mutation(mutation, genome=genome)

    assert result.allowed is False
    assert "workflow step ids must be unique" in result.reason


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
