from stemos.genome.loader import load_default_genome
from stemos.genome.models import EnvironmentSpec, Genome
from stemos.harness.builder import HarnessBuilder
from stemos.scenarios.loader import load_scenario


def test_default_genome_schema_includes_environment():
    genome = load_default_genome()

    assert isinstance(genome, Genome)
    assert isinstance(genome.environment, EnvironmentSpec)
    assert genome.environment.cleanup_policy == "keep_run_artifacts"
    assert genome.roles[0].name == "Founder"


def test_harness_builder_materializes_genome_environment(tmp_path):
    genome = load_default_genome()
    genome.environment.required_artifacts = ["artifacts/final_output.md"]
    genome.environment.file_templates = {"artifacts/final_output.md": "# Final Output\n"}
    scenario = load_scenario("scenarios/toy_structured_answer").scenario

    harness = HarnessBuilder().materialize(
        genome, scenario, workspace_dir=tmp_path / "workspace"
    )

    assert "Founder" in harness.roles
    assert harness.workflow[0].id == "understand_task"
    assert harness.tool_registry.names() == ["call_model"]
    assert (tmp_path / "workspace" / "artifacts" / "final_output.md").exists()
