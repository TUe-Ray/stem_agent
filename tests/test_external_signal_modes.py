import pytest

from stem_agent.evolution.loop import EvolutionLoop
from stem_agent.kernel.evaluator import EvaluationResult


def _result(score: float) -> EvaluationResult:
    return EvaluationResult(score=score, promotion_score=score, train_score=score)


def test_internal_only_ignores_external_score():
    loop = EvolutionLoop()
    result = loop._combine_external_signal(internal_result=_result(0.5), external_result=_result(0.9), signal_mode="internal_only", external_signal_weight=0.25)
    assert result.promotion_score == pytest.approx(0.5)


def test_internal_plus_external_score_combines_scores():
    loop = EvolutionLoop()
    result = loop._combine_external_signal(internal_result=_result(0.5), external_result=_result(0.9), signal_mode="internal_plus_external_score", external_signal_weight=0.25)
    assert result.promotion_score == pytest.approx(0.6)
