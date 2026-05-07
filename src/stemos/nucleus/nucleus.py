from __future__ import annotations

import random

from stemos.nucleus.operators import (
    CrossoverMutation,
    FirstOrderMutation,
    HyperMutation,
    MutationOperator,
    ZeroOrderMutation,
)


def select_operator(archive, stagnation_count: int) -> MutationOperator:
    if stagnation_count >= 5:
        return HyperMutation()
    if stagnation_count >= 3:
        return ZeroOrderMutation()
    if stagnation_count >= 1 and len(archive) >= 3 and random.random() < 0.25:
        return CrossoverMutation()
    return FirstOrderMutation()
