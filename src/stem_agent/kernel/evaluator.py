from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from stem_agent.genome.models import Genome
from stem_agent.harness.runner import HarnessRunResult
from stem_agent.llm.client import build_guardian_system_prompt
from stem_agent.nucleus.model_client import ModelClient
from stem_agent.scenarios.schema import Scenario


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
    task_quality: float = 0.0
    cost_efficiency: float = 0.0
    safety_score: float = 1.0
    stability_score: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    case_results: list[CaseEvaluation] = Field(default_factory=list)
    cost_estimate: float = 0.0
    complexity_penalty: float = 0.0
    split_policy: str = "train_only"


class GuardianFitnessEvaluator:
    """Immutable promotion evaluator. Nucleus is not allowed to mutate this."""

    def __init__(self, model_client: ModelClient | None = None):
        self.model_client = model_client or ModelClient()

    def configure_run(self, run_dir: str | Path | None) -> None:
        self.model_client.configure_run(run_dir)

    def evaluate(
        self,
        genome: Genome,
        scenario: Scenario,
        train_runs: list[HarnessRunResult],
        validation_runs: list[HarnessRunResult] | None = None,
        *,
        run_dir: str | Path | None = None,
    ) -> EvaluationResult:
        if run_dir is not None:
            self.configure_run(run_dir)
        validation_runs = validation_runs or []
        self._audit_guardian_call(scenario, train_runs, validation_runs)
        complexity_penalty = self._complexity_penalty(genome)
        train_cases = [
            self._evaluate_case(genome, scenario, run, complexity_penalty)
            for run in train_runs
        ]
        validation_cases = [
            self._evaluate_case(genome, scenario, run, complexity_penalty)
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
            promotion_score = (0.40 * train_score) + (0.60 * validation_score)
            split_policy = "weighted_train_validation_40_60"

        task_quality = promotion_score
        safety_score = 0.0 if any(run.blocked for run in train_runs + validation_runs) else 1.0
        cost_efficiency = self._clamp(1.0 - min(sum(run.cost_estimate for run in train_runs + validation_runs) / 5.0, 1.0))
        stability_score = self._stability_score(train_cases, validation_cases)

        all_cases = train_cases + validation_cases
        metrics = self._mean_metrics([case.metrics for case in all_cases])
        failures = [failure for case in all_cases for failure in case.failures]
        cost_estimate = sum(run.cost_estimate for run in train_runs + validation_runs)

        return EvaluationResult(
            score=promotion_score,
            promotion_score=promotion_score,
            train_score=train_score,
            validation_score=validation_score,
            task_quality=task_quality,
            cost_efficiency=cost_efficiency,
            safety_score=safety_score,
            stability_score=stability_score,
            metrics=metrics,
            failures=failures,
            case_results=all_cases,
            cost_estimate=cost_estimate,
            complexity_penalty=complexity_penalty,
            split_policy=split_policy,
        )


    def _stability_score(
        self,
        train_cases: list[CaseEvaluation],
        validation_cases: list[CaseEvaluation],
    ) -> float:
        case_scores = [case.score for case in train_cases + validation_cases]
        if not case_scores:
            return 1.0
        mean_score = self._mean(case_scores)
        variance = sum((score - mean_score) ** 2 for score in case_scores) / len(case_scores)
        return self._clamp(1.0 - variance)

    def _evaluate_case(
        self,
        genome: Genome,
        scenario: Scenario,
        run: HarnessRunResult,
        complexity_penalty: float,
    ) -> CaseEvaluation:
        output = run.final_output or ""
        if scenario.evaluation_criteria:
            return self._evaluate_case_with_criteria(scenario, run, output, complexity_penalty)
        requirement_coverage, failures = self._requirement_coverage(
            scenario.expected_output.requirements, output
        )
        scenario_success = self._scenario_success(scenario, output)
        format_validity = self._format_validity(output)
        artifact_presence = self._artifact_presence(run)
        self_review_usage = self._self_review_usage(run, output)
        workflow_completion = self._workflow_completion(genome, run)
        quality_gate_usage = self._quality_gate_usage(run)
        generated_tool_usage = self._generated_tool_usage(run)
        cost_penalty = min(run.cost_estimate / 5.0, 1.0)

        raw_score = (
            0.28 * requirement_coverage
            + 0.14 * format_validity
            + 0.12 * artifact_presence
            + 0.16 * self_review_usage
            + 0.14 * workflow_completion
            + 0.08 * quality_gate_usage
            + 0.08 * generated_tool_usage
            - 0.03 * cost_penalty
            - 0.05 * complexity_penalty
        )
        if run.blocked:
            raw_score -= 0.15
        score = self._clamp(raw_score)
        metrics = {
            "requirement_coverage": requirement_coverage,
            "scenario_success": scenario_success,
            "format_validity": format_validity,
            "artifact_presence": artifact_presence,
            "self_review_usage": self_review_usage,
            "workflow_completion": workflow_completion,
            "quality_gate_usage": quality_gate_usage,
            "generated_tool_usage": generated_tool_usage,
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

    def _evaluate_case_with_criteria(
        self,
        scenario: Scenario,
        run: HarnessRunResult,
        output: str,
        complexity_penalty: float,
    ) -> CaseEvaluation:
        metrics: dict[str, float] = {}
        failures: list[str] = []
        total_weight = sum(max(float(item.weight), 0.0) for item in scenario.evaluation_criteria) or 1.0
        weighted_score = 0.0
        for criterion in scenario.evaluation_criteria:
            value = self._criterion_score(criterion.method, output, run, criterion.pattern)
            metrics[criterion.name] = value
            weighted_score += (max(float(criterion.weight), 0.0) / total_weight) * value
            if value <= 0.0:
                failures.append(f"Failed criterion: {criterion.name}")
        score = self._clamp(weighted_score - 0.05 * complexity_penalty)
        metrics["complexity_penalty"] = complexity_penalty
        return CaseEvaluation(
            case_id=run.case_id,
            score=score,
            metrics=metrics,
            failures=failures,
            final_output=output,
        )

    def _criterion_score(
        self,
        method: str,
        output: str,
        run: HarnessRunResult,
        pattern: str | None,
    ) -> float:
        if method == "exact_match_after_extraction":
            from stem_agent.benchmarks.gsm8k import exact_match_after_extraction

            return exact_match_after_extraction(output, run.expected_output or "")
        if method == "regex_check":
            expression = pattern or r"\d+.*[\+\-\*\/].*\d+"
            return 1.0 if re.search(expression, output, re.DOTALL) else 0.0
        if method == "llm_judge":
            return self._no_hallucinated_numbers_score(output, run)
        return 0.0

    def _no_hallucinated_numbers_score(self, output: str, run: HarnessRunResult) -> float:
        problem_text = json.dumps(run.case_input or {}, sort_keys=True)
        allowed = set(re.findall(r"-?\d+\.?\d*", problem_text))
        if run.expected_output:
            allowed.add(str(run.expected_output))
        produced = re.findall(r"-?\d+\.?\d*", output)
        if not produced:
            return 0.0
        extra = [number for number in produced if number not in allowed]
        return 1.0 if len(extra) <= max(2, len(produced) // 2) else 0.0

    def _audit_guardian_call(
        self,
        scenario: Scenario,
        train_runs: list[HarnessRunResult],
        validation_runs: list[HarnessRunResult],
    ) -> None:
        criteria = [item.model_dump(mode="json") for item in scenario.evaluation_criteria]
        if not criteria:
            criteria = [
                {"name": item.name, "weight": item.weight, "description": item.description}
                for item in scenario.success_criteria
            ]
        if not criteria:
            criteria = [
                {"name": "expected_output_requirement", "weight": 1.0, "description": item}
                for item in scenario.expected_output.requirements
            ]
        payload = {
            "evaluation_criteria": criteria,
            "outputs": [
                {
                    "case_id": run.case_id,
                    "split": "train" if run in train_runs else "validation",
                    "final_output": run.final_output,
                }
                for run in train_runs + validation_runs
            ],
        }
        prompt = json.dumps(payload, sort_keys=True)
        system_prompt = build_guardian_system_prompt(criteria)
        response = {"criterion_scores": {}, "overall": 0.0, "reasoning": "guardian evaluator audit"}
        self.model_client.audit_call(
            role="guardian",
            system_prompt=system_prompt,
            prompt=prompt,
            response=response,
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

    def _artifact_presence(self, run: HarnessRunResult) -> float:
        for trace in run.traces:
            if trace.get("event") == "environment_materialized" and trace.get("artifacts"):
                return 1.0
        output = run.final_output.lower()
        artifact_markers = ["acceptance criteria", "qa report", "decision log", "draft output"]
        return min(sum(1 for marker in artifact_markers if marker in output) / 3.0, 1.0)

    def _self_review_usage(self, run: HarnessRunResult, output: str) -> float:
        output_lower = output.lower()
        score = 0.0
        if "check the draft" in output_lower or "quality review" in output_lower:
            score += 0.25
        if "review notes applied" in output_lower:
            score += 0.25
        if any(trace.get("step_id") == "review_against_requirements" for trace in run.traces):
            score += 0.25
        if any(trace.get("event") == "quality_gate" and trace.get("passed") for trace in run.traces):
            score += 0.25
        return self._clamp(score)

    def _workflow_completion(self, genome: Genome, run: HarnessRunResult) -> float:
        workflow_steps = [trace for trace in run.traces if trace.get("event") == "workflow_step"]
        if not genome.workflow:
            return 0.0
        completed_ratio = len(workflow_steps) / len(genome.workflow)
        score = min(completed_ratio, 1.0) * 0.35
        completed_ids = {str(trace.get("step_id")) for trace in workflow_steps}
        if "review_against_requirements" in completed_ids:
            score += 0.25
        if "revise_final_output" in completed_ids:
            score += 0.30
        if run.final_output == run.outputs.get("final_output", ""):
            score += 0.10
        return self._clamp(score)

    def _quality_gate_usage(self, run: HarnessRunResult) -> float:
        gate_traces = [trace for trace in run.traces if trace.get("event") == "quality_gate"]
        if not gate_traces:
            return 0.0
        passed = sum(1 for trace in gate_traces if trace.get("passed"))
        return passed / len(gate_traces)

    def _generated_tool_usage(self, run: HarnessRunResult) -> float:
        for trace in run.traces:
            if trace.get("event") == "quality_gate" and trace.get("generated_tool"):
                return 1.0
        return 0.0

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
