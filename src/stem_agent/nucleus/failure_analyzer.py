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
        patterns.extend(self._metric_deficit_patterns(evaluation.metrics))
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

    def _metric_deficit_patterns(self, metrics: dict[str, float]) -> list[FailurePattern]:
        patterns: list[FailurePattern] = []
        checks = [
            ("Low self-review usage", "self_review_usage", "review_loop_missing", "add_review_step", 0.60),
            ("Low quality-gate usage", "quality_gate_usage", "quality_gate_missing", "add_quality_gate", 0.60),
            ("Low artifact presence", "artifact_presence", "artifact_gap", "modify_environment", 0.50),
            ("Low actionability", "actionability", "weak_actionability", "add_revision_step", 0.65),
            ("Low input specificity", "input_specificity", "weak_input_specificity", "add_revision_step", 0.65),
        ]
        for label, metric_name, category, operator, threshold in checks:
            value = float(metrics.get(metric_name, 1.0))
            if value < threshold:
                patterns.append(
                    FailurePattern(
                        kind=f"{label}: {metric_name}={value:.3f}",
                        count=1,
                        severity=round(threshold - value, 4),
                        category=category,
                        suggested_operator=operator,
                    )
                )
        complexity = float(metrics.get("complexity_penalty", 0.0))
        if complexity >= 0.35:
            patterns.append(
                FailurePattern(
                    kind=f"High complexity penalty: complexity_penalty={complexity:.3f}",
                    count=1,
                    severity=round(complexity, 4),
                    category="complexity_pressure",
                    suggested_operator="simplify_genome",
                )
            )
        return patterns[:5]
