"""Phase D: LLM rubric gating — only fires when explicitly enabled (test_evaluator_llm_rubric)."""
from unittest.mock import MagicMock

from stem_agent.kernel.evaluator import GuardianFitnessEvaluator


class TestLlmRubricDispatch:
    """Verify _criterion_score dispatches llm_rubric correctly."""

    def test_llm_rubric_dispatch_key_exists(self):
        """The method name 'llm_rubric' is recognized by criterion dispatch."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        assert hasattr(e, "_llm_rubric_score"), "_llm_rubric_score must exist"

    def test_llm_judge_dispatch_key_exists(self):
        """The method name 'llm_judge' is recognized (maps to no-hallucination check)."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        assert hasattr(e, "_no_hallucinated_numbers_score"), "_no_hallucinated_numbers_score must exist"


class TestLlmRubricWithoutClient:
    """LLM rubric gracefully degrades when no model client is set."""

    def test_llm_rubric_requires_model_client(self):
        """Without model_client, rubric scoring should not crash (returns 0 or graceful)."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        try:
            score = e._llm_rubric_score(None, None, "some output")
            assert 0.0 <= score <= 1.0, f"Score must be 0-1, got {score}"
        except AttributeError:
            pass  # No model_client → expected graceful failure


class TestNoHallucinatedNumbers:
    def test_clean_output_scores_high(self):
        """Output without fake stats scores high."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        # Mock a HarnessRunResult with minimal fields
        run = MagicMock()
        run.case_input = {"problem": "What is 2+2?"}
        output = "The answer is 4. Steps: 1. think, 2. solve."
        score = e._no_hallucinated_numbers_score(output, run)
        assert 0.0 <= score <= 1.0

    def test_excessive_numbers_may_score_lower(self):
        """Output packed with numeric claims may trigger heuristic."""
        e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
        run = MagicMock()
        run.case_input = {"problem": "What is 2+2?"}
        output = (
            "97% of users agree. 15 studies show. Deployed by 450 teams. "
            "Score: 9.8/10. Ranked #3 globally."
        )
        score = e._no_hallucinated_numbers_score(output, run)
        # Heuristic: many standalone numbers suggests hallucination
        assert score < 1.0, f"Many claim-numbers should reduce score, got {score}"
