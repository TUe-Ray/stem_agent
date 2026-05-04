from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("STEMOS_MODEL", "gpt-5.5"))
    max_cost_usd: float = Field(
        default_factory=lambda: float(os.getenv("STEMOS_MAX_COST_USD", "2.0"))
    )
    workspace_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("STEMOS_WORKSPACE_DIR", "workspace"))
    )
    offline_mode: bool = Field(
        default_factory=lambda: os.getenv("STEMOS_OFFLINE_MODE", "true").lower()
        in {"1", "true", "yes", "on"}
    )


def load_settings() -> Settings:
    return Settings()
