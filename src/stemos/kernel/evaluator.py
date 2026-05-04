from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from stemos.genome.models import Genome
from stemos.harness.runner import HarnessRunResult
from stemos.scenarios.schema import Scenario


class CaseEvaluation(BaseModel):
    case_id: str
    score: float
    metrics: dict[str, float]
    failures: list[str] = Field(default_factory=list)
    final_output: str = ""


class EvaluationResult(BaseModel):
    score: float
    promotion_score: float
    train_score: float
    validation_score: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    case_results: list[CaseEvaluation] = Field(default_factory=list)
    cost_estimate: float = 0.0
    complexity_penalty: float = 0.0
    split_policy: str = "train_only"


class GuardianFitnessEvaluator:
    """Immutable promotion evaluator. Nucleus is not allowed to mutate this."""

    def evaluate(
        self,
        genome: Genome,
        scenario: Scenario,
        train_runs: list[HarnessRunResult],
        validation_runs: list[HarnessRunResult] | None = None,
    ) -> EvaluationResult:
        validation_runs = validation_runs or []
        complexity_penalty = self._complexity_penalty(genome)
        train_cases = [
            self._evaluate_case(scenario, run, complexity_penalty) for run in train_runs
        ]
        validation_cases = [
            self._evaluate_case(scenario, run, complexity_penalty)
            for run in validation_runs
        ]
        train_score = self._mean([case.score for case in train_cases])
        validation_score = (
            self._mean([case.score for case in validation_cases])
            if validation_cases
            else None
        )
        if validation_score is None:
            promotion_score = train_score
            split_policy = "train_only"
        else:
            promotion_score = (0.35 * train_score) + (0.65 * validation_score)
            split_policy = "weighted_train_validation_35_65"

        all_cases = train_cases + validation_cases
        metrics = self._mean_metrics([case.metrics for case in all_cases])
        failures = [failure for case in all_cases for failure in case.failures]
        cost_estimate = sum(run.cost_estimate for run in train_runs + validation_runs)

        return EvaluationResult(
            score=promotion_score,
            promotion_score=promotion_score,
            train_score=train_score,
            validation_score=validation_score,
            metrics=metrics,
            failures=failures,
            case_results=all_cases,
            cost_estimate=cost_estimate,
            complexity_penalty=complexity_penalty,
            split_policy=split_policy,
        )

    def _evaluate_case(
        self,
        scenario: Scenario,
        run: HarnessRunResult,
        complexity_penalty: float,
    ) -> CaseEvaluation:
        output = run.final_output or ""
        requirement_coverage, failures = self._requirement_coverage(
            scenario.expected_output.requirements, output
        )
        scenario_success = self._scenario_success(scenario, output)
        format_validity = self._format_validity(output)
        robustness = 0.0 if run.blocked else self._robustness(scenario, output)
        cost_penalty = min(run.cost_estimate / 5.0, 1.0)

        raw_score = (
            0.40 * requirement_coverage
            + 0.25 * scenario_success
            + 0.20 * format_validity
            + 0.10 * robustness
            - 0.05 * cost_penalty
            - 0.05 * complexity_penalty
        )
        score = self._clamp(raw_score)
        metrics = {
            "requirement_coverage": requirement_coverage,
            "scenario_success": scenario_success,
            "format_validity": format_validity,
            "robustness": robustness,
            "cost_penalty": cost_penalty,
            "complexity_penalty": complexity_penalty,
        }
        if run.blocked:
            failures.append("Candidate harness was blocked by a quality gate")
        return CaseEvaluation(
            case_id=run.case_id,
            score=score,
            metrics=metrics,
            failures=failures,
            final_output=output,
        )

    def _requirement_coverage(
        self, requirements: list[str], output: str
    ) -> tuple[float, list[str]]:
        if not requirements:
            return 1.0, []
        passed = 0
        failures: list[str] = []
        for requirement in requirements:
            if self._requirement_satisfied(requirement, output):
                passed += 1
            else:
                failures.append(f"Missing requirement: {requirement}")
        return passed / len(requirements), failures

    def _requirement_satisfied(self, requirement: str, output: str) -> bool:
        req = requirement.lower()
        out = output.lower()
        checks = {
            "summary": ["summary"],
            "step": ["steps", "1.", "- "],
            "final": ["final answer", "final output", "recommendation"],
            "acceptance": ["acceptance criteria", "acceptance"],
            "qa": ["qa report", "quality review", "review"],
            "decision": ["decision log", "decision"],
            "artifact": ["artifact", ".md"],
        }
        for key, needles in checks.items():
            if key in req:
                return any(needle in out for needle in needles)

        words = [
            word
            for word in re.findall(r"[a-zA-Z]{4,}", req)
            if word not in {"must", "include", "with", "that", "this", "output"}
        ]
        if not words:
            return True
        return sum(1 for word in words if word in out) >= max(1, len(words) // 2)

    def _scenario_success(self, scenario: Scenario, output: str) -> float:
        out = output.lower()
        score = 0.5
        if len(output.strip()) >= 80:
            score += 0.2
        if any(token in out for token in ["step", "action", "acceptance", "final"]):
            score += 0.2
        if "?" not in output or "Do not ask unnecessary" not in " ".join(scenario.constraints):
            score += 0.1
        return self._clamp(score)

    def _format_validity(self, output: str) -> float:
        if not output.strip():
            return 0.0
        score = 0.3
        if re.search(r"(^|\n)#{1,3} ", output):
            score += 0.25
        if re.search(r"(^|\n)(\d+\.|- )", output):
            score += 0.2
        if "summary" in output.lower():
            score += 0.15
        if "final" in output.lower():
            score += 0.1
        return self._clamp(score)

    def _robustness(self, scenario: Scenario, output: str) -> float:
        out = output.lower()
        score = 0.6
        if "cannot" not in out and "i don't" not in out:
            score += 0.2
        if len(output.split()) >= 25:
            score += 0.2
        return self._clamp(score)

    def _complexity_penalty(self, genome: Genome) -> float:
        generated_tools = genome.tools.get("generated", []) or []
        size = (
            max(len(genome.roles) - 1, 0) * 0.12
            + max(len(genome.workflow) - 2, 0) * 0.08
            + len(generated_tools) * 0.15
            + len(genome.quality_gates) * 0.08
            + len(genome.environment.required_artifacts) * 0.03
        )
        return self._clamp(size)

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _mean_metrics(self, metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        keys = sorted({key for item in metrics for key in item})
        return {key: self._mean([item.get(key, 0.0) for item in metrics]) for key in keys}

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
