from __future__ import annotations

from pydantic import BaseModel


class BudgetTracker(BaseModel):
    max_cost_usd: float
    spent_usd: float = 0.0

    @property
    def remaining_usd(self) -> float:
        return max(self.max_cost_usd - self.spent_usd, 0.0)

    def can_spend(self, amount: float) -> bool:
        return self.spent_usd + amount <= self.max_cost_usd

    def record(self, amount: float) -> None:
        if not self.can_spend(amount):
            raise RuntimeError("StemOS budget exceeded")
        self.spent_usd += amount
