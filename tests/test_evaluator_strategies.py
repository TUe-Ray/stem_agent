"""Tests for evaluator strategy pattern and LLM judge evaluator."""
import pytest
from unittest.mock import MagicMock, patch

from stem_agent.kernel.evaluator_strategies import (
    EvalMetrics,
    EvaluatorStrategy,
    LLMJudgeEvaluator,
)
from stem_agent.kernel.evaluator import GuardianFitnessEvaluator
from stem_agent.kernel.evaluator_weights import EvaluatorWeights
from stem_agent.genome.models import Genome
from stem_agent.harness.runner import HarnessRunResult
from stem_agent.scenarios.schema import Scenario


def make_minimal_genome() -> Genome:
    return Genome.model_validate({
        "genome_version": 0, "name": "test", "scenario_name": "test",
        "task_diagnosis": {}, "roles": [], "workflow": [],
        "tools": {"builtin": [], "generated": []}, "memory": {},
        "quality_gates": [], "self_evaluation": {"rubric": ""},
        "retry_policy": {"max_attempts": 1, "revise_on_failure": False},
        "stop_rule": {}, "environment": {
            "workspace_layout": [], "required_artifacts": [],
            "artifact_purpose": {}, "file_templates": {},
            "cleanup_policy": "keep_run_artifacts",
        },
    })


def make_minimal_scenario() -> Scenario:
    return Scenario.model_validate({
        "scenario": {"name": "test", "description": "test", "task_class": "test"},
        "input_format": {"type": "text", "fields": []},
        "expected_output": {"type": "text", "requirements": []},
        "constraints": [], "available_builtin_tools": [],
        "signal_policy": {"layer_1_enabled": True, "expose_aggregate_score": True,
                          "expose_score_delta": True, "expose_direction": True},
        "evolution": {"max_generations": 1, "patience": 1, "min_delta": 0.01,
                       "max_mutations_per_generation": 1, "signal_mode": "internal_only"},
    })


def make_minimal_run(output="test output") -> HarnessRunResult:
    return HarnessRunResult(
        case_id="case_1",
        final_output=output,
        traces=[],
        cost_estimate=0.001,
        case_input={"task": "test task"},
    )


class TestLLMJudgeEvaluator:
    def test_name(self):
        judge = LLMJudgeEvaluator()
        assert judge.name() == "llm_judge"

    def test_parse_valid_json(self):
        response = '{"requirement_match": 0.7, "content_quality": 0.6, "structure": 0.8, "weaknesses": ["vague"], "summary": "ok"}'
        result = LLMJudgeEvaluator._parse_judge_response(response)
        assert result["requirement_match"] == 0.7
        assert result["content_quality"] == 0.6

    def test_parse_markdown_wrapped_json(self):
        response = '```json\n{"requirement_match": 0.6, "content_quality": 0.7, "structure": 0.9, "weaknesses": [], "summary": "decent"}\n```'
        result = LLMJudgeEvaluator._parse_judge_response(response)
        assert result["requirement_match"] == 0.6

    def test_evaluate_case_with_mock_client(self):
        """Test evaluate_case with a mocked model client."""
        judge = LLMJudgeEvaluator()
        mock_client = MagicMock()
        mock_client.call.return_value = (
            '{"requirement_match": 0.85, "content_quality": 0.75, "structure": 0.9, "weaknesses": ["vague step 2"], "summary": "solid"}'
        )
        judge.model_client = mock_client

        genome = make_minimal_genome()
        scenario = make_minimal_scenario()
        run = make_minimal_run("The answer is 42. Steps: 1. think, 2. solve.")

        metrics = judge.evaluate_case(genome, scenario, run, 0.0)
        assert 0.0 <= metrics.score <= 1.0
        assert "requirement_match" in metrics.metrics
        assert metrics.metrics["requirement_match"] == 0.85
        mock_client.call.assert_called_once()

    def test_evaluate_case_fallback_on_error(self):
        """When judge call fails, fallback to conservative 0.5 scores."""
        judge = LLMJudgeEvaluator()
        mock_client = MagicMock()
        mock_client.call.side_effect = RuntimeError("API error")
        judge.model_client = mock_client

        genome = make_minimal_genome()
        scenario = make_minimal_scenario()
        run = make_minimal_run("some output")

        metrics = judge.evaluate_case(genome, scenario, run, 0.0)
        # Fallback should produce a reasonable score
        assert 0.4 <= metrics.score <= 0.6
        assert "requirement_match" in metrics.metrics


class TestEvaluatorWithStrategy:
    def test_heuristic_default_no_strategy(self):
        """Without strategy, uses built-in heuristic (backward compat)."""
        e = GuardianFitnessEvaluator(weights=EvaluatorWeights())
        genome = make_minimal_genome()
        scenario = make_minimal_scenario()
        run = make_minimal_run("## Summary\n\nStep 1. Do it.\n\nFinal answer: done.")

        result = e._evaluate_case(genome, scenario, run, 0.0)
        assert 0.0 <= result.score <= 1.0
        assert result.final_output is not None

    def test_with_llm_judge_strategy(self):
        """With LLM judge strategy, delegates to the strategy."""
        judge = LLMJudgeEvaluator()
        mock_client = MagicMock()
        mock_client.call.return_value = (
            '{"requirement_match": 0.9, "content_quality": 0.9, "structure": 0.85, "weaknesses": [], "summary": "great"}'
        )
        judge.model_client = mock_client

        e = GuardianFitnessEvaluator(strategy=judge)
        genome = make_minimal_genome()
        scenario = make_minimal_scenario()
        run = make_minimal_run("## Summary\n\nFinal answer: 42.")

        result = e._evaluate_case(genome, scenario, run, 0.0)
        assert 0.0 <= result.score <= 1.0
        assert "requirement_match" in result.metrics
        mock_client.call.assert_called_once()
