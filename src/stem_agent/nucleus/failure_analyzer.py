from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from stem_agent.kernel.evaluator import EvaluationResult


class FailurePattern(BaseModel):
    kind: str
    count: int
    severity: float


class FailureAnalyzer:
    def analyze_patterns(self, evaluation: EvaluationResult) -> list[FailurePattern]:
        if not evaluation.failures:
            return []
        counts = Counter(evaluation.failures)
        total = max(sum(counts.values()), 1)
        patterns: list[FailurePattern] = []
        for failure, count in counts.most_common(5):
            patterns.append(
                FailurePattern(
                    kind=failure,
                    count=count,
                    severity=round(count / total, 4),
                )
            )
        return patterns

    def analyze(self, evaluation: EvaluationResult) -> list[str]:
        return [pattern.kind for pattern in self.analyze_patterns(evaluation)]
