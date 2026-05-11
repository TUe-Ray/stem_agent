from __future__ import annotations

from pydantic import BaseModel


class EvaluatorWeights(BaseModel):
    requirement_coverage: float = 0.20
    format_validity: float = 0.08
    constraint_adherence: float = 0.10
    actionability: float = 0.16
    input_specificity: float = 0.14
    artifact_presence: float = 0.08
    self_review_usage: float = 0.14
    workflow_completion: float = 0.08
    quality_gate_usage: float = 0.10
    generated_tool_usage: float = 0.04
    cost_penalty: float = 0.03
    complexity_penalty: float = 0.05

    def metric_weights(self) -> dict[str, float]:
        return {
            "requirement_coverage": self.requirement_coverage,
            "format_validity": self.format_validity,
            "constraint_adherence": self.constraint_adherence,
            "actionability": self.actionability,
            "input_specificity": self.input_specificity,
            "artifact_presence": self.artifact_presence,
            "self_review_usage": self.self_review_usage,
            "workflow_completion": self.workflow_completion,
            "quality_gate_usage": self.quality_gate_usage,
            "generated_tool_usage": self.generated_tool_usage,
        }

    def normalized(self) -> "EvaluatorWeights":
        positive = self.metric_weights()
        total = sum(max(value, 0.0) for value in positive.values()) or 1.0
        update = {key: max(value, 0.0) / total for key, value in positive.items()}
        return self.model_copy(update=update)
