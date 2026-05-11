__all__ = ["EvaluationResult", "Guardian", "GuardianFitnessEvaluator", "ValidationResult"]


def __getattr__(name: str):
    if name == "EvaluationResult":
        from stem_agent.kernel.evaluator import EvaluationResult

        return EvaluationResult
    if name == "GuardianFitnessEvaluator":
        from stem_agent.kernel.evaluator import GuardianFitnessEvaluator

        return GuardianFitnessEvaluator
    if name == "Guardian":
        from stem_agent.kernel.guardian import Guardian

        return Guardian
    if name == "ValidationResult":
        from stem_agent.kernel.validators import ValidationResult

        return ValidationResult
    raise AttributeError(name)
