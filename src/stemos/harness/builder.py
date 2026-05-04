from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stemos.genome.models import EnvironmentSpec, Genome, RoleSpec, WorkflowStep
from stemos.harness.tool_registry import ToolRegistry
from stemos.scenarios.schema import Scenario


@dataclass
class MaterializedHarness:
    genome: Genome
    scenario: Scenario
    roles: dict[str, RoleSpec]
    workflow: list[WorkflowStep]
    tool_registry: ToolRegistry
    memory_layout: dict
    quality_gates: list
    retry_policy: dict
    environment: EnvironmentSpec
    workspace_dir: Path


class HarnessBuilder:
    """Materializes a genome into an executable operating harness."""

    def materialize(
        self,
        genome: Genome,
        scenario: Scenario,
        *,
        workspace_dir: str | Path,
    ) -> MaterializedHarness:
        workspace = Path(workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        self._materialize_environment(genome.environment, workspace)
        return MaterializedHarness(
            genome=genome,
            scenario=scenario,
            roles={role.name: role for role in genome.roles},
            workflow=list(genome.workflow),
            tool_registry=ToolRegistry.from_genome(genome),
            memory_layout=dict(genome.memory),
            quality_gates=list(genome.quality_gates),
            retry_policy=dict(genome.retry_policy),
            environment=genome.environment,
            workspace_dir=workspace,
        )

    def _materialize_environment(self, environment: EnvironmentSpec, workspace: Path) -> None:
        for item in environment.workspace_layout:
            self._safe_child(workspace, item).mkdir(parents=True, exist_ok=True)

        for artifact in environment.required_artifacts:
            path = self._safe_child(workspace, artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            template = environment.file_templates.get(artifact, "")
            if not path.exists():
                path.write_text(template, encoding="utf-8")

    def _safe_child(self, root: Path, relative: str) -> Path:
        child = (root / relative).resolve()
        root_resolved = root.resolve()
        if root_resolved != child and root_resolved not in child.parents:
            raise ValueError(f"Environment path escapes workspace: {relative}")
        return child
