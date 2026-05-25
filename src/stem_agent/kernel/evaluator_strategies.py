"""Pluggable evaluator strategies for stem_agent.

Strategy pattern: replaceable evaluation logic behind a stable interface.
EvolutionLoop picks a strategy based on scenario config or env var.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from stem_agent.genome.models import Genome
from stem_agent.harness.runner import HarnessRunResult
from stem_agent.scenarios.schema import Scenario


@dataclass
class EvalMetrics:
    """Per-case evaluation metrics produced by any evaluator strategy."""
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    final_output: str = ""


class EvaluatorStrategy(ABC):
    """Abstract evaluator — scores a single harness run against a scenario."""

    @abstractmethod
    def evaluate_case(
        self,
        genome: Genome,
        scenario: Scenario,
        run: HarnessRunResult,
        complexity_penalty: float,
    ) -> EvalMetrics:
        """Score one case. Returns EvalMetrics with 0-1 score."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name for config/logging."""


class LLMJudgeEvaluator(EvaluatorStrategy):
    """LLM-as-judge: send (task, output, rubric) to a judge model for 0-1 scoring.

    Uses a lightweight, cheap model (gpt-4o-mini or equivalent) to evaluate
    output quality on correctness, completeness, clarity, and actionability.
    """

    RUBRIC_PROMPT = """You are an expert evaluator judging the quality of an AI agent's output.
Rate the output on these criteria from 0.0 (terrible) to 1.0 (perfect):

1. **correctness** (0-1): Did the output correctly address the task? Are facts/answers accurate?
2. **completeness** (0-1): Did it cover all requirements? Any missing sections or steps?
3. **clarity** (0-1): Is the output well-structured, readable, and free of confusion?
4. **actionability** (0-1): Can a human follow this output to take action? Are steps concrete?

Return ONLY a JSON object:
{"correctness": 0.X, "completeness": 0.X, "clarity": 0.X, "actionability": 0.X, "summary": "one-line verdict"}"""

    def __init__(self, model_client=None):
        from stem_agent.nucleus.model_client import ModelClient
        self.model_client = model_client or ModelClient()

    def name(self) -> str:
        return "llm_judge"

    def evaluate_case(
        self,
        genome: Genome,
        scenario: Scenario,
        run: HarnessRunResult,
        complexity_penalty: float,
    ) -> EvalMetrics:
        # Build the evaluation prompt
        task_input = json.dumps(run.case_input or {}, sort_keys=True)
        expected = run.expected_output or ""
        reference = json.dumps(getattr(run, 'reference_notes', '') or '', sort_keys=True)

        prompt = (
            f"TASK INPUT:\n{task_input[:1500]}\n\n"
            f"EXPECTED OUTPUT / REFERENCE:\n{expected[:800]}\n{reference[:800]}\n\n"
            f"AGENT OUTPUT TO EVALUATE:\n{run.final_output[:3000]}\n"
        )

        try:
            response = self.model_client.call(
                prompt,
                system_prompt=self.RUBRIC_PROMPT,
                temperature=0.0,
            )
            result = self._parse_judge_response(str(response))
        except Exception:
            # Fallback: if judge call fails, score conservatively
            result = {"correctness": 0.5, "completeness": 0.5, "clarity": 0.5, "actionability": 0.5}

        # Weighted blend → final score
        score = (
            0.35 * result.get("correctness", 0.5)
            + 0.25 * result.get("completeness", 0.5)
            + 0.20 * result.get("clarity", 0.5)
            + 0.20 * result.get("actionability", 0.5)
        )
        score = max(0.0, min(1.0, score))

        metrics = {
            "correctness": result.get("correctness", 0.5),
            "completeness": result.get("completeness", 0.5),
            "clarity": result.get("clarity", 0.5),
            "actionability": result.get("actionability", 0.5),
            "judge_summary": result.get("summary", ""),
        }

        failures = []
        if result.get("correctness", 1.0) < 0.4:
            failures.append("Low correctness (judge)")
        if result.get("completeness", 1.0) < 0.4:
            failures.append("Low completeness (judge)")

        return EvalMetrics(
            score=score,
            metrics=metrics,
            failures=failures,
            final_output=run.final_output,
        )

    @staticmethod
    def _parse_judge_response(response: str) -> dict:
        """Extract JSON from judge response (may be wrapped in markdown or prose)."""
        import re
        # Try to find JSON block
        m = re.search(r'\{[^{}]*"correctness"[^{}]*\}', response, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        # Fallback: try parsing whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}
