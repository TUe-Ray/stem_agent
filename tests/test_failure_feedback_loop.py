from stem_agent.kernel.evaluator import EvaluationResult
from stem_agent.nucleus.failure_analyzer import FailureAnalyzer


def test_failure_pattern_classification():
    evaluation = EvaluationResult(
        score=0.2,
        train_score=0.2,
        validation_score=0.0,
        promotion_score=0.2,
        failures=["Missing requirement: final answer", "Missing requirement: final answer"],
    )
    patterns = FailureAnalyzer().analyze_patterns(evaluation)
    assert patterns[0].kind == "Missing requirement: final answer"
    assert patterns[0].category == "missing_required_section"
    assert patterns[0].suggested_operator == "add_quality_gate"
    assert patterns[0].count == 2
