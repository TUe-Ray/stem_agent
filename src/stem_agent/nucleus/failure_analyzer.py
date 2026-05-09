from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from stem_agent.kernel.evaluator import EvaluationResult


class FailurePattern(BaseModel):
    kind: str
    count: int
    severity: float
    category: str = "unknown"
    suggested_operator: str | None = None


class FailureAnalyzer:
    def analyze_patterns(self, evaluation: EvaluationResult) -> list[FailurePattern]:
        if not evaluation.failures:
            return []
        counts = Counter(evaluation.failures)
        total = max(sum(counts.values()), 1)
        patterns: list[FailurePattern] = []
        for failure, count in counts.most_common(5):
            category, suggested_operator = self._classify(failure)
            patterns.append(
                FailurePattern(
                    kind=failure,
                    count=count,
                    severity=round(count / total, 4),
                    category=category,
                    suggested_operator=suggested_operator,
                )
            )
        return patterns

    def analyze(self, evaluation: EvaluationResult) -> list[str]:
        return [pattern.kind for pattern in self.analyze_patterns(evaluation)]

    def _classify(self, failure: str) -> tuple[str, str | None]:
        lowered = failure.lower()
        if "missing requirement" in lowered or "required section" in lowered:
            return "missing_required_section", "add_quality_gate"
        if "exact_match" in lowered or "arithmetic" in lowered or "number" in lowered:
            return "reasoning_or_calculation_error", "add_verification_step"
        if "blocked by a quality gate" in lowered:
            return "quality_gate_block", "edit_quality_gate"
        if "format" in lowered or "schema" in lowered:
            return "format_or_schema_error", "add_quality_gate"
        if "cost" in lowered or "budget" in lowered:
            return "cost_pressure", "modify_retry_policy"
        return "unknown", None
