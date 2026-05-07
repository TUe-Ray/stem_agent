from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ValidationResult(BaseModel):
    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls, reason: str = "allowed") -> "ValidationResult":
        return cls(allowed=True, reason=reason)

    @classmethod
    def reject(cls, reason: str) -> "ValidationResult":
        return cls(allowed=False, reason=reason)


class MutationSafetyValidator:
    def __init__(self, run_dir: str | Path | None = None):
        self.run_dir = Path(run_dir) if run_dir else None

    def validate_self_eval_rubric_change(
        self,
        old_genome: dict,
        new_genome: dict,
        scenario: dict,
    ) -> ValidationResult:
        old_rubric = self._rubric(old_genome)
        new_rubric = self._rubric(new_genome)
        if old_rubric == new_rubric:
            return ValidationResult.allow("self-evaluation rubric unchanged")

        scenario_criteria = self._scenario_criteria(scenario)
        missing = [
            criterion
            for criterion in scenario_criteria
            if not self._criterion_appears(criterion, new_rubric)
        ]
        max_delta = self._max_reference_score_delta(old_rubric, new_rubric, scenario)
        if missing:
            reason = "Self-evaluation rubric removed scenario criteria: " + "; ".join(missing)
            self._log_rubric_audit(old_rubric, new_rubric, scenario_criteria, missing, max_delta, "rejected", reason)
            return ValidationResult.reject(reason)
        if max_delta > 0.15:
            reason = f"Self-evaluation rubric inflates reference scores by {max_delta:.3f}"
            self._log_rubric_audit(old_rubric, new_rubric, scenario_criteria, missing, max_delta, "rejected", reason)
            return ValidationResult.reject(reason)

        self._log_rubric_audit(
            old_rubric,
            new_rubric,
            scenario_criteria,
            missing,
            max_delta,
            "allowed",
            "self-evaluation rubric preserves scenario criteria",
        )
        return ValidationResult.allow("self-evaluation rubric preserves scenario criteria")

    def validate_stop_rule_change(
        self,
        old_genome: dict,
        new_genome: dict,
    ) -> ValidationResult:
        old_rule = dict(old_genome.get("stop_rule") or {})
        new_rule = dict(new_genome.get("stop_rule") or {})
        if old_rule == new_rule:
            return ValidationResult.allow("stop_rule unchanged")

        allowed_fields = {"max_generations", "score_threshold", "patience"}
        changed = {key for key in set(old_rule) | set(new_rule) if old_rule.get(key) != new_rule.get(key)}
        extra = changed - allowed_fields
        if extra:
            return ValidationResult.reject(
                "Unsupported stop_rule mutation fields: " + ", ".join(sorted(extra))
            )

        if "max_generations" in changed:
            if int(new_rule.get("max_generations", 0)) < int(old_rule.get("max_generations", 0)):
                return ValidationResult.reject("stop_rule.max_generations may only increase")
        if "patience" in changed:
            if int(new_rule.get("patience", 0)) < int(old_rule.get("patience", 0)):
                return ValidationResult.reject("stop_rule.patience may only increase")
        if "score_threshold" in changed:
            threshold = float(new_rule.get("score_threshold", 0.0))
            if threshold < 0.5 or threshold > 0.99:
                return ValidationResult.reject("stop_rule.score_threshold must be between 0.5 and 0.99")
        return ValidationResult.allow("stop_rule changes are monotonic and bounded")

    def _rubric(self, genome: dict) -> list[str]:
        rubric = (genome.get("self_evaluation") or {}).get("rubric", [])
        if isinstance(rubric, str):
            return [rubric] if rubric.strip() else []
        if isinstance(rubric, dict):
            return [f"{key}: {value}" for key, value in rubric.items()]
        return [str(item) for item in (rubric or [])]

    def _scenario_criteria(self, scenario: dict) -> list[str]:
        criteria = scenario.get("evaluation_criteria") or []
        extracted: list[str] = []
        for item in criteria:
            if isinstance(item, dict):
                parts = [str(item.get("name", ""))]
                if item.get("description"):
                    parts.append(str(item["description"]))
                extracted.append(" ".join(part for part in parts if part).strip())
            else:
                extracted.append(str(item))
        if extracted:
            return [item for item in extracted if item]

        expected = scenario.get("expected_output") or {}
        requirements = expected.get("requirements") or []
        if requirements:
            return [str(item) for item in requirements]

        success = scenario.get("success_criteria") or []
        for item in success:
            if isinstance(item, dict):
                extracted.append(str(item.get("description") or item.get("name") or ""))
        return [item for item in extracted if item]

    def _criterion_appears(self, criterion: str, rubric: list[str]) -> bool:
        if not criterion.strip():
            return True
        rubric_text = " ".join(rubric).lower()
        if criterion.lower() in rubric_text:
            return True
        words = self._important_words(criterion)
        if not words:
            return True
        matches = sum(1 for word in words if word in rubric_text)
        return matches / len(words) >= 0.6

    def _max_reference_score_delta(
        self,
        old_rubric: list[str],
        new_rubric: list[str],
        scenario: dict,
    ) -> float:
        train_cases = scenario.get("train_cases") or []
        deltas: list[float] = []
        for case in train_cases[:3]:
            output = self._reference_output(case)
            old_score = self._score_output_with_rubric(output, old_rubric)
            new_score = self._score_output_with_rubric(output, new_rubric)
            deltas.append(new_score - old_score)
        return max(deltas, default=0.0)

    def _reference_output(self, case: Any) -> str:
        if not isinstance(case, dict):
            return str(case)
        for key in ["reference_output", "expected_output", "reference_notes", "answer"]:
            if case.get(key):
                return str(case[key])
        return json.dumps(case.get("input", case), sort_keys=True)

    def _score_output_with_rubric(self, output: str, rubric: list[str]) -> float:
        if not rubric:
            return 1.0
        passed = 0
        output_lower = output.lower()
        for item in rubric:
            words = self._important_words(item)
            if not words:
                passed += 1
                continue
            matches = sum(1 for word in words if word in output_lower)
            if matches / len(words) >= 0.5:
                passed += 1
        return passed / len(rubric)

    def _important_words(self, text: str) -> list[str]:
        stopwords = {
            "must",
            "include",
            "with",
            "that",
            "this",
            "output",
            "score",
            "criterion",
            "criteria",
            "method",
        }
        return [
            word
            for word in re.findall(r"[a-zA-Z0-9_]{4,}", text.lower())
            if word not in stopwords
        ]

    def _log_rubric_audit(
        self,
        old_rubric: list[str],
        new_rubric: list[str],
        scenario_criteria: list[str],
        missing: list[str],
        max_delta: float,
        decision: str,
        reason: str,
    ) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        diff = list(
            difflib.unified_diff(
                old_rubric,
                new_rubric,
                fromfile="old_rubric",
                tofile="new_rubric",
                lineterm="",
            )
        )
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_rubric": old_rubric,
            "new_rubric": new_rubric,
            "scenario_criteria": scenario_criteria,
            "missing_criteria": missing,
            "max_reference_score_delta": round(max_delta, 6),
            "decision": decision,
            "reason": reason,
            "diff": diff,
        }
        with (self.run_dir / "rubric_audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
