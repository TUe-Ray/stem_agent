from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel


ToolCallable = Callable[[dict[str, Any]], dict[str, Any]]


class Tool(BaseModel):
    name: str
    description: str
    run: ToolCallable

    model_config = {"arbitrary_types_allowed": True}
