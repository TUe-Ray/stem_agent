from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


MutationType = Literal[
    "add_role",
    "edit_role",
    "add_workflow_step",
    "edit_workflow_step",
    "remove_workflow_step",
    "create_tool",
    "edit_tool",
    "add_quality_gate",
    "edit_quality_gate",
    "modify_memory_schema",
    "modify_retry_policy",
    "modify_self_evaluation",
    "modify_stop_rule",
    "modify_environment",
]


class MutationProposal(BaseModel):
    mutation_type: MutationType
    target: str
    rationale: str
    expected_improvement: str
    risk: str
    patch: dict[str, Any] = Field(default_factory=dict)


class MutationPlan(BaseModel):
    summary: str
    failure_patterns: list[str] = Field(default_factory=list)
    proposed_mutations: list[MutationProposal] = Field(default_factory=list)


class TaskDiagnosis(BaseModel):
    task_type: str
    expected_task_solving_pattern: str
    likely_failure_modes: list[str] = Field(default_factory=list)
    likely_needed_capabilities: list[str] = Field(default_factory=list)
    initial_architecture_hypothesis: str
    initial_evaluation_hypothesis: str
