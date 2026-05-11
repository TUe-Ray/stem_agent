from stem_agent.kernel.evaluator_calibrator import CalibrationSample, EvaluatorCalibrator
from stem_agent.kernel.evaluator_weights import EvaluatorWeights


def test_calibrator_respects_max_weight_delta():
    old = EvaluatorWeights()
    new, _record = EvaluatorCalibrator().calibrate(
        current_weights=old,
        samples=[
            CalibrationSample(metrics={"requirement_coverage": 0.1, "format_validity": 0.9}, external_score=0.2),
            CalibrationSample(metrics={"requirement_coverage": 0.9, "format_validity": 0.2}, external_score=0.8),
        ],
        generation=1,
        max_weight_delta=0.03,
    )
    for key, old_val in old.model_dump().items():
        assert abs(new.model_dump()[key] - old_val) <= 0.05
