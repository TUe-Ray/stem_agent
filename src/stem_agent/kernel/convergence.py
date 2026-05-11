from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ConvergencePolicy(BaseModel):
    plateau_window: int = 5
    plateau_epsilon: float = 0.01
    hidden_eval_stability_window: int = 3
    hidden_eval_stability_epsilon: float = 0.01
    absolute_score_threshold: float = 0.9
    max_generations: int = 20
    budget_token_limit: int = 200000


class ConvergenceDecision(BaseModel):
    stop: bool
    reason: str
    action: Literal[
        "continue",
        "force_zero_order",
        "freeze_best",
        "stop_max_generations",
        "stop_budget",
    ]
    plateau_detected: bool


class ConvergenceEngine:
    def __init__(
        self,
        policy: ConvergencePolicy | dict | None = None,
        *,
        run_dir: str | Path | None = None,
        hidden_baseline_score: float | None = None,
    ):
        self.policy = (
            policy
            if isinstance(policy, ConvergencePolicy)
            else ConvergencePolicy.model_validate(policy or {})
        )
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.hidden_baseline_score = hidden_baseline_score
        self.validation_scores_per_generation: list[float] = []
        self.hidden_eval_scores_per_generation: list[float | None] = []
        self.total_tokens_used = 0
        self.current_generation = 0
        self.current_zero_order_was_attempted = False
        self.plateau_detected = False
        self.zero_order_attempted_after_plateau = False
        self._plateau_baseline_score: float | None = None
        self._last_decision = ConvergenceDecision(
            stop=False,
            reason="not evaluated yet",
            action="continue",
            plateau_detected=False,
        )

    def update(
        self,
        *,
        generation_number: int,
        validation_score: float,
        hidden_eval_score: float | None,
        tokens_used_this_generation: int,
        zero_order_was_attempted: bool,
    ) -> None:
        self.current_generation = generation_number
        self.validation_scores_per_generation.append(validation_score)
        self.hidden_eval_scores_per_generation.append(hidden_eval_score)
        self.total_tokens_used += max(int(tokens_used_this_generation), 0)
        self.current_zero_order_was_attempted = zero_order_was_attempted
        if self.plateau_detected and zero_order_was_attempted:
            self.zero_order_attempted_after_plateau = True

    def evaluate(self) -> ConvergenceDecision:
        policy = self.policy
        validation_score = self.validation_scores_per_generation[-1]
        hidden_score = self.hidden_eval_scores_per_generation[-1]
        criteria_votes: dict[str, bool] = {}

        if self.current_generation + 1 >= policy.max_generations:
            decision = ConvergenceDecision(
                stop=True,
                action="stop_max_generations",
                reason="maximum generation hard cap reached",
                plateau_detected=self.plateau_detected,
            )
            self._log(decision, criteria_votes)
            return decision

        if self.total_tokens_used >= policy.budget_token_limit:
            decision = ConvergenceDecision(
                stop=True,
                action="stop_budget",
                reason="token budget hard cap reached",
                plateau_detected=self.plateau_detected,
            )
            self._log(decision, criteria_votes)
            return decision

        hidden_acceptable = self._hidden_eval_acceptable(hidden_score)
        criteria_votes["hidden_eval_acceptable"] = hidden_acceptable
        if validation_score >= policy.absolute_score_threshold and hidden_acceptable:
            decision = ConvergenceDecision(
                stop=True,
                action="freeze_best",
                reason="absolute validation threshold reached and hidden eval is acceptable",
                plateau_detected=self.plateau_detected,
            )
            self._log(decision, criteria_votes)
            return decision

        if self.plateau_detected and self.zero_order_attempted_after_plateau:
            improved = self._improved_after_plateau()
            criteria_votes["zero_order_improved_after_plateau"] = improved
            if improved:
                self.plateau_detected = False
                self.zero_order_attempted_after_plateau = False
                self._plateau_baseline_score = None
                decision = ConvergenceDecision(
                    stop=False,
                    action="continue",
                    reason="zero-order mutation improved beyond plateau epsilon",
                    plateau_detected=False,
                )
                self._log(decision, criteria_votes)
                return decision
            hidden_stable = self._hidden_eval_stable()
            criteria_votes["hidden_eval_stable"] = hidden_stable
            if hidden_stable:
                decision = ConvergenceDecision(
                    stop=True,
                    action="freeze_best",
                    reason="plateau persisted after zero-order mutation and convergence criteria passed",
                    plateau_detected=True,
                )
                self._log(decision, criteria_votes)
                return decision
            decision = ConvergenceDecision(
                stop=False,
                action="continue",
                reason="plateau persisted but convergence votes are incomplete",
                plateau_detected=True,
            )
            self._log(decision, criteria_votes)
            return decision

        if self._plateau_now():
            self.plateau_detected = True
            self._plateau_baseline_score = validation_score
            decision = ConvergenceDecision(
                stop=False,
                action="force_zero_order",
                reason="validation score plateau detected; force zero-order mutation next",
                plateau_detected=True,
            )
            self._log(decision, criteria_votes)
            return decision

        decision = ConvergenceDecision(
            stop=False,
            action="continue",
            reason="convergence criteria not met",
            plateau_detected=self.plateau_detected,
        )
        self._log(decision, criteria_votes)
        return decision

    def _hidden_eval_acceptable(self, hidden_score: float | None) -> bool:
        if hidden_score is None or hidden_score == 0.0:
            return True
        if self.hidden_baseline_score is None:
            return True
        return hidden_score >= self.hidden_baseline_score - 0.10

    def _hidden_eval_stable(self) -> bool:
        real_scores = [score for score in self.hidden_eval_scores_per_generation if score is not None and score != 0.0]
        window = self.policy.hidden_eval_stability_window
        if len(real_scores) < window:
            return False
        recent = real_scores[-window:]
        return max(recent) - min(recent) < self.policy.hidden_eval_stability_epsilon

    def _plateau_now(self) -> bool:
        window = self.policy.plateau_window
        if len(self.validation_scores_per_generation) < window:
            return False
        recent = self.validation_scores_per_generation[-window:]
        return max(recent) - min(recent) < self.policy.plateau_epsilon

    def _improved_after_plateau(self) -> bool:
        if self._plateau_baseline_score is None:
            return False
        return (
            self.validation_scores_per_generation[-1] - self._plateau_baseline_score
            > self.policy.plateau_epsilon
        )

    def _log(self, decision: ConvergenceDecision, criteria_votes: dict[str, bool]) -> None:
        self._last_decision = decision
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        item = {
            "generation_number": self.current_generation,
            "validation_score_history": self.validation_scores_per_generation,
            "hidden_eval_score_history": self.hidden_eval_scores_per_generation,
            "total_tokens_used": self.total_tokens_used,
            "plateau_detected": decision.plateau_detected,
            "zero_order_attempted_after_plateau": self.zero_order_attempted_after_plateau,
            "criteria_votes": criteria_votes,
            "stop": decision.stop,
            "action": decision.action,
            "reason": decision.reason,
        }
        with (self.run_dir / "convergence_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
