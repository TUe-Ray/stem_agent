from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stem_agent.kernel.evaluator_weights import EvaluatorWeights


class CalibrationSample(BaseModel):
    metrics: dict[str, float]
    external_score: float


class CalibrationRecord(BaseModel):
    generation: int
    old_weights: dict[str, float]
    new_weights: dict[str, float]
    correlations: dict[str, float] = Field(default_factory=dict)
    max_weight_delta: float
    sample_count: int
    reason: str


class EvaluatorCalibrator:
    def calibrate(self, *, current_weights: EvaluatorWeights, samples: list[CalibrationSample], generation: int, max_weight_delta: float = 0.03) -> tuple[EvaluatorWeights, CalibrationRecord]:
        if len(samples) < 2:
            rec = CalibrationRecord(generation=generation, old_weights=current_weights.model_dump(mode="json"), new_weights=current_weights.model_dump(mode="json"), correlations={}, max_weight_delta=max_weight_delta, sample_count=len(samples), reason="Not enough samples for calibration.")
            return current_weights, rec
        metric_names = sorted({k for s in samples for k in s.metrics})
        ext = [s.external_score for s in samples]
        corr: dict[str, float] = {}
        new = dict(current_weights.model_dump(mode="json"))
        old = dict(new)
        for metric in metric_names:
            values = [float(s.metrics.get(metric, 0.0)) for s in samples]
            c = self._correlation(values, ext)
            corr[metric] = c
            if metric in new:
                delta = max(-max_weight_delta, min(max_weight_delta, c * max_weight_delta))
                new[metric] = max(0.0, float(new[metric]) + delta)
        nw = EvaluatorWeights.model_validate(new).normalized()
        rec = CalibrationRecord(generation=generation, old_weights=old, new_weights=nw.model_dump(mode="json"), correlations=corr, max_weight_delta=max_weight_delta, sample_count=len(samples), reason="Adjusted internal evaluator weights based on correlation with external dev benchmark score.")
        return nw, rec

    def _correlation(self, xs: list[float], ys: list[float]) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            return 0.0
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = sum((x - mx) ** 2 for x in xs) ** 0.5
        deny = sum((y - my) ** 2 for y in ys) ** 0.5
        if denx == 0 or deny == 0:
            return 0.0
        return max(-1.0, min(1.0, num / (denx * deny)))
