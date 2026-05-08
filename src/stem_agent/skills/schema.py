from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SkillType = Literal[
    "workflow_step",
    "quality_gate",
    "tool",
    "role_instruction",
    "retry_policy",
    "self_eval_rubric_fragment",
    "memory_schema",
]


class SkillReuseRecord(BaseModel):
    run_id: str
    scenario_name: str
    generations_on_probation: int = 0
    observed_score_lift: float | None = None
    evicted: bool = False


class AtomicSkill(BaseModel):
    id: str
    name: str
    description: str
    skill_type: SkillType
    genome_fragment: dict[str, Any]
    domain_tags: list[str] = Field(default_factory=list)
    origin_run_id: str
    origin_scenario_name: str
    origin_generation: int
    score_lift: float
    reuse_history: list[SkillReuseRecord] = Field(default_factory=list)
