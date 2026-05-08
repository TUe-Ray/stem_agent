from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


class Settings(BaseModel):
    model: str = Field(default_factory=lambda: _env("STEM_AGENT_MODEL", "gpt-4.1-mini"))
    openai_endpoint: str = Field(
        default_factory=lambda: _env("STEM_AGENT_OPENAI_ENDPOINT", "auto")
    )
    max_cost_usd: float = Field(
        default_factory=lambda: float(_env("STEM_AGENT_MAX_COST_USD", "2.0"))
    )
    workspace_dir: Path = Field(
        default_factory=lambda: Path(_env("STEM_AGENT_WORKSPACE_DIR", "workspace"))
    )
    test_mode: bool = False


def load_settings() -> Settings:
    return Settings()
