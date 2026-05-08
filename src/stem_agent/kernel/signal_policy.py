from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SignalLayer(Enum):
    LAYER_0 = 0
    LAYER_1 = 1
    LAYER_2 = 2


@dataclass
class NucleusSignal:
    """
    The only object Nucleus is allowed to receive from Guardian.
    """

    generation: int
    mutation_type: str
    direction: Literal["improved", "degraded", "neutral", "unknown"]
    aggregate_score: float | None = None
    previous_aggregate_score: float | None = None
    score_delta: float | None = None


@dataclass
class SignalPolicy:
    """
    Controls which aggregate evaluation signals are exposed to Nucleus.
    """

    layer_1_enabled: bool = True
    expose_aggregate_score: bool = True
    expose_score_delta: bool = True
    expose_direction: bool = True

    def build_nucleus_signal(
        self,
        generation: int,
        mutation_type: str,
        current_score: float,
        previous_score: float,
    ) -> NucleusSignal:
        direction: Literal["improved", "degraded", "neutral", "unknown"] = (
            "improved"
            if current_score > previous_score + 0.01
            else "degraded"
            if current_score < previous_score - 0.01
            else "neutral"
        )
        return NucleusSignal(
            generation=generation,
            mutation_type=mutation_type,
            direction=direction if self.expose_direction else "unknown",
            aggregate_score=(
                current_score
                if self.layer_1_enabled and self.expose_aggregate_score
                else None
            ),
            previous_aggregate_score=(
                previous_score
                if self.layer_1_enabled and self.expose_aggregate_score
                else None
            ),
            score_delta=(
                round(current_score - previous_score, 4)
                if self.layer_1_enabled and self.expose_score_delta
                else None
            ),
        )

    def assert_no_layer2_leak(self, nucleus_prompt: str) -> None:
        forbidden_patterns = [
            "validation_case",
            "expected_output",
            "ground_truth",
            "criterion_score",
            "per_case",
            "hidden_eval",
        ]
        prompt_lower = nucleus_prompt.lower()
        for pattern in forbidden_patterns:
            if pattern.lower() in prompt_lower:
                raise ValueError(
                    "SignalPolicy violation: Layer 2 content detected in Nucleus prompt. "
                    f"Pattern: '{pattern}'. Nucleus must never see validation case details."
                )
