from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ScenarioMeta(BaseModel):
    name: str
    description: str
    task_class: str


class InputField(BaseModel):
    name: str
    required: bool = True


class InputFormat(BaseModel):
    type: str = "text"
    fields: list[InputField] = Field(default_factory=list)


class ExpectedOutput(BaseModel):
    type: str = "markdown"
    requirements: list[str] = Field(default_factory=list)


class SuccessCriterion(BaseModel):
    name: str
    weight: float
    description: str


class EvaluationCriterion(BaseModel):
    name: str
    weight: float
    method: str
    description: str = ""
    pattern: str | None = None


class EvolutionConfig(BaseModel):
    max_generations: int = 5
    patience: int = 2
    min_delta: float = 0.03
    max_mutations_per_generation: int = 4
    max_cost_usd: float = 2.0
    max_workflow_steps: int = 8
    max_roles: int = 6
    max_environment_artifacts: int = 8


class Scenario(BaseModel):
    scenario: ScenarioMeta
    input_format: InputFormat
    expected_output: ExpectedOutput
    constraints: list[str] = Field(default_factory=list)
    available_builtin_tools: list[str] = Field(default_factory=list)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)

    @property
    def name(self) -> str:
        return self.scenario.name

    @field_validator("success_criteria")
    @classmethod
    def weights_are_reasonable(
        cls, criteria: list[SuccessCriterion]
    ) -> list[SuccessCriterion]:
        if criteria:
            total = sum(item.weight for item in criteria)
            if total <= 0:
                raise ValueError("success_criteria weights must sum to a positive value")
        return criteria


class TaskCase(BaseModel):
    id: str = ""
    input: dict[str, Any]
    reference_notes: str = ""
    expected_output: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_case(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        raw_input = normalized.get("input", {})
        if isinstance(raw_input, str):
            normalized["input"] = {"user_request": raw_input, "problem": raw_input}
        if not normalized.get("id"):
            basis = str(normalized.get("input", ""))[:80]
            digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:8]
            normalized["id"] = f"case_{digest}"
        if normalized.get("expected_output") is not None:
            normalized["expected_output"] = str(normalized["expected_output"])
        return normalized
