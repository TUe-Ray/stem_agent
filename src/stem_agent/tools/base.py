from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field


ToolCallable = Callable[[dict[str, Any]], dict[str, Any]]

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}


class Tool(BaseModel):
    name: str
    description: str
    run: ToolCallable
    parameters: dict[str, Any] = Field(default_factory=lambda: dict(_DEFAULT_PARAMETERS))

    model_config = {"arbitrary_types_allowed": True}

    def to_api_dict(self) -> dict[str, Any]:
        """Return an OpenAI Responses-API-compatible function tool dict."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
