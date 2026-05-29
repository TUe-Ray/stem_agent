from __future__ import annotations

from pydantic import BaseModel


class EvaluatorWeights(BaseModel):
    """Weights for the stem_agent evaluator.

    Design principles:
    - task_completion dominates (50%) — measured by scenario-specific criteria
    - process metrics are secondary (26%) — how the agent worked
    - innovation rewards useful mutations (12%) — generated tools, safeguards
    - penalties discourage bloat (12%) — cost, complexity

    No more regex keyword-spotting for core scores.
    """

    # ── Task Outcome (62%) — what actually matters ──
    task_completion: float = 0.50       # Did the agent solve the problem?
    constraint_adherence: float = 0.08  # Did it follow the rules?
    format_validity: float = 0.04       # Is the output well-formed?

    # ── Process Quality (26%) — how the agent worked ──
    workflow_completion: float = 0.08   # Did it complete all steps?
    diagnosis_quality: float = 0.06     # Did it understand the problem?
    architecture_fit: float = 0.06      # Is the genome well-structured?
    actionability: float = 0.06         # Is the output directly usable?

    # ── Innovation (12%) — evolution-specific ──
    generated_tool_usage: float = 0.08  # Did generated tools actually help?
    safeguard_effectiveness: float = 0.04  # Do quality gates catch errors?

    # ── Penalties ──
    cost_penalty: float = 0.05          # Token cost
    complexity_penalty: float = 0.07    # Genome bloat

    # ── Aggregate (used by LLM judge strategies) ──
    output_quality: float = 0.65
    stem_process_quality: float = 0.20

    # ── Deprecated / zeroed out (kept for backward compat) ──
    requirement_coverage: float = 0.0   # → replaced by task_completion
    input_specificity: float = 0.0      # → text-matching, not a real signal
    reference_alignment: float = 0.0    # → text-matching, not a real signal
    artifact_presence: float = 0.0      # → binary, not useful
    self_review_usage: float = 0.0      # → keyword-spotting spam magnet
    quality_gate_usage: float = 0.0     # → folded into safeguard_effectiveness
    minimality_score: float = 0.0       # → folded into complexity_penalty

    def metric_weights(self) -> dict[str, float]:
        return {
            "task_completion": self.task_completion,
            "constraint_adherence": self.constraint_adherence,
            "format_validity": self.format_validity,
            "workflow_completion": self.workflow_completion,
            "diagnosis_quality": self.diagnosis_quality,
            "architecture_fit": self.architecture_fit,
            "actionability": self.actionability,
            "generated_tool_usage": self.generated_tool_usage,
            "safeguard_effectiveness": self.safeguard_effectiveness,
        }

    def normalized(self) -> "EvaluatorWeights":
        positive = self.metric_weights()
        total = sum(max(value, 0.0) for value in positive.values()) or 1.0
        update = {key: max(value, 0.0) / total for key, value in positive.items()}
        return self.model_copy(update=update)
