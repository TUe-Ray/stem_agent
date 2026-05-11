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


def test_metric_deficits_create_actionable_feedback_patterns():
    evaluation = EvaluationResult(
        score=0.4,
        train_score=0.4,
        validation_score=0.4,
        promotion_score=0.4,
        metrics={
            "self_review_usage": 0.0,
            "quality_gate_usage": 0.0,
            "actionability": 0.3,
            "input_specificity": 0.4,
        },
    )

    patterns = FailureAnalyzer().analyze_patterns(evaluation)
    categories = {pattern.category for pattern in patterns}

    assert "review_loop_missing" in categories
    assert "quality_gate_missing" in categories
    assert "weak_actionability" in categories
