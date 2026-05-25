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

    RUBRIC_PROMPT = """You are a strict evaluator. Your job is to find weaknesses in AI agent outputs, not to be nice.
Be critical. Most outputs should score between 0.4 and 0.8. Reserve 0.9+ for truly exceptional answers.

Score the output against the REFERENCE/EXPECTED answer on these criteria:

1. **requirement_match** (0-1): How many of the explicit requirements does the output satisfy?
   Compare against the EXPECTED OUTPUT requirements one by one.
   - 1.0: ALL requirements met with specific, accurate content
   - 0.7: Most requirements met, minor gaps
   - 0.4: Several requirements missing or wrong
   - 0.1: Barely addresses the task

2. **content_quality** (0-1): Is the content accurate, specific, and non-generic?
   - 1.0: Specific, accurate, domain-appropriate detail
   - 0.7: Mostly correct but somewhat generic
   - 0.4: Vague, generic advice that could apply to any task
   - 0.1: Factually wrong or nonsensical

3. **structure** (0-1): Is the output well-organized with clear sections?
   - 1.0: Excellent structure with logical flow and appropriate formatting
   - 0.5: Adequate structure but could be better organized
   - 0.1: Wall of text, no organization

4. **weaknesses** (list): List 1-3 specific weaknesses or missing elements.
   If you cannot find any weaknesses, you are not looking hard enough.

Return ONLY a JSON object:
{"requirement_match": 0.X, "content_quality": 0.X, "structure": 0.X, "weaknesses": ["specific weakness 1", "..."], "summary": "one-line verdict"}"""

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
            result = {"requirement_match": 0.5, "content_quality": 0.5, "structure": 0.5}

        # Weighted blend → final score (requirement_match is most important)
        score = (
            0.40 * result.get("requirement_match", 0.5)
            + 0.35 * result.get("content_quality", 0.5)
            + 0.25 * result.get("structure", 0.5)
        )
        score = max(0.0, min(1.0, score))

        metrics = {
            "requirement_match": result.get("requirement_match", 0.5),
            "content_quality": result.get("content_quality", 0.5),
            "structure": result.get("structure", 0.5),
            "judge_summary": result.get("summary", ""),
            "judge_weaknesses": result.get("weaknesses", []),
        }

        failures = []
        if result.get("requirement_match", 1.0) < 0.4:
            failures.append("Low requirement match (judge)")
        if result.get("content_quality", 1.0) < 0.4:
            failures.append("Low content quality (judge)")
        # Include judge's own weaknesses as failure signals
        weaknesses = result.get("weaknesses", [])
        if isinstance(weaknesses, list):
            failures.extend(weaknesses[:3])

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
        # Try to find JSON block: match { ... } with balanced braces
        # Simple approach: find the outermost { } pair
        start = response.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(response)):
                if response[i] == "{":
                    depth += 1
                elif response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start:i + 1])
                        except json.JSONDecodeError:
                            break
        # Fallback: try parsing whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}
