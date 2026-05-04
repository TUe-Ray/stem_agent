from __future__ import annotations

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
