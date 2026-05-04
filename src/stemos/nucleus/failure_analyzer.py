from __future__ import annotations

from collections import Counter

from stemos.kernel.evaluator import EvaluationResult


class FailureAnalyzer:
    def analyze(self, evaluation: EvaluationResult) -> list[str]:
        if not evaluation.failures:
            return []
        counts = Counter(evaluation.failures)
        return [failure for failure, _ in counts.most_common(5)]
