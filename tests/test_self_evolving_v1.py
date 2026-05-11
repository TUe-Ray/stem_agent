from __future__ import annotations

from stem_agent.kernel.convergence import ConvergenceEngine, ConvergencePolicy


def test_convergence_plateau_forces_zero_order_before_freeze_and_logs(tmp_path):
    engine = ConvergenceEngine(
        ConvergencePolicy(
            plateau_window=3,
            plateau_epsilon=0.01,
            max_generations=10,
            absolute_score_threshold=0.99,
            hidden_eval_stability_window=3,
            hidden_eval_stability_epsilon=0.01,
        ),
        run_dir=tmp_path,
        hidden_baseline_score=0.5,
    )

    decisions = []
    for generation in range(3):
        engine.update(
            generation_number=generation,
            validation_score=0.5,
            hidden_eval_score=0.5,
            tokens_used_this_generation=10,
            zero_order_was_attempted=False,
        )
        decisions.append(engine.evaluate())

    assert decisions[-1].action == "force_zero_order"
    assert decisions[-1].stop is False

    engine.update(
        generation_number=3,
        validation_score=0.5,
        hidden_eval_score=0.5,
        tokens_used_this_generation=10,
        zero_order_was_attempted=True,
    )
    decision = engine.evaluate()

    assert decision.action == "freeze_best"
    assert decision.stop is True
    assert (tmp_path / "convergence_log.jsonl").read_text(encoding="utf-8").count("\n") == 4


def test_convergence_hard_caps_cannot_be_overridden(tmp_path):
    engine = ConvergenceEngine(
        ConvergencePolicy(max_generations=2, budget_token_limit=1_000_000),
        run_dir=tmp_path / "max",
    )
    for generation in range(2):
        engine.update(
            generation_number=generation,
            validation_score=0.1,
            hidden_eval_score=None,
            tokens_used_this_generation=1,
            zero_order_was_attempted=False,
        )
        decision = engine.evaluate()
    assert decision.action == "stop_max_generations"
    assert decision.stop is True

    budget_engine = ConvergenceEngine(
        ConvergencePolicy(max_generations=20, budget_token_limit=10),
        run_dir=tmp_path / "budget",
    )
    budget_engine.update(
        generation_number=0,
        validation_score=0.99,
        hidden_eval_score=None,
        tokens_used_this_generation=10,
        zero_order_was_attempted=False,
    )
    budget_decision = budget_engine.evaluate()
    assert budget_decision.action == "stop_budget"
