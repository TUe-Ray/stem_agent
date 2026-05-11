from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AllowedMutationSignal(BaseModel):
    internal_score: float
    external_dev_score: float | None = None
    calibrated_internal_score: float | None = None
    combined_score: float | None = None
    score_delta: float | None = None
    signal_mode: str = "internal_only"
    external_signal_weight: float = 0.0
    fitness_vector: dict[str, float] = Field(default_factory=dict)
    failure_categories: list[str] = Field(default_factory=list)

    def nucleus_safe_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluatorPrivateContext(BaseModel):
    train_case_count: int = 0
    validation_case_count: int = 0
    external_benchmark_case_count: int = 0
    final_holdout_case_count: int = 0
