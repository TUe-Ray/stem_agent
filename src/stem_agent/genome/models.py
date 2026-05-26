from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RoleSpec(BaseModel):
    name: str
    description: str
    instructions: str
    allowed_tools: list[str] = Field(default_factory=list)
    temperature: float | None = None  # None = use default (0.0)


class WorkflowStep(BaseModel):
    id: str
    role: str
    action: str
    input_from: list[str] = Field(default_factory=list)
    output_key: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    kind: Literal["builtin", "generated"]
    path: str | None = None
    test_path: str | None = None


class QualityGate(BaseModel):
    name: str
    description: str
    check_type: Literal["llm_judge", "python", "schema", "manual_placeholder"]
    required: bool = True


class EnvironmentSpec(BaseModel):
    workspace_layout: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    artifact_purpose: dict[str, str] = Field(default_factory=dict)
    file_templates: dict[str, str] = Field(default_factory=dict)
    cleanup_policy: str = "keep_run_artifacts"


class Genome(BaseModel):
    genome_version: int
    name: str
    scenario_name: str | None = None
    task_diagnosis: dict[str, Any] = Field(default_factory=dict)
    roles: list[RoleSpec]
    workflow: list[WorkflowStep]
    tools: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    quality_gates: list[QualityGate] = Field(default_factory=list)
    self_evaluation: dict[str, Any] = Field(default_factory=dict)
    retry_policy: dict[str, Any] = Field(default_factory=dict)
    stop_rule: dict[str, Any] = Field(default_factory=dict)
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    intra_test_reflection_enabled: bool = False
    max_intra_reflection_retries: int = 1

    @field_validator("roles")
    @classmethod
    def role_names_are_unique(cls, roles: list[RoleSpec]) -> list[RoleSpec]:
        names = [role.name for role in roles]
        if len(names) != len(set(names)):
            raise ValueError("role names must be unique")
        return roles

    @field_validator("workflow")
    @classmethod
    def workflow_ids_are_unique(cls, steps: list[WorkflowStep]) -> list[WorkflowStep]:
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("workflow step ids must be unique")
        return steps
