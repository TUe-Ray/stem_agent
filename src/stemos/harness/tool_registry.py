from __future__ import annotations

from pathlib import Path

from stemos.genome.models import Genome
from stemos.tools.base import Tool
from stemos.tools.builtins import builtin_tools
from stemos.tools.generated_loader import load_generated_tool


class ToolRegistry:
    def __init__(self, tools: dict[str, Tool]):
        self._tools = tools

    @classmethod
    def from_genome(cls, genome: Genome) -> "ToolRegistry":
        available = builtin_tools()
        selected: dict[str, Tool] = {}
        for name in genome.tools.get("builtin", []) or []:
            if name in available:
                selected[name] = available[name]

        for spec in genome.tools.get("generated", []) or []:
            if isinstance(spec, dict):
                name = spec.get("name")
                path = spec.get("path")
                description = spec.get("description", "")
            else:
                name = spec.name
                path = spec.path
                description = spec.description
            if name and path and Path(path).exists():
                selected[name] = Tool(
                    name=name,
                    description=description,
                    run=load_generated_tool(path),
                )
        return cls(selected)

    def get(self, names: list[str]) -> list[Tool]:
        return [self._tools[name] for name in names if name in self._tools]

    def names(self) -> list[str]:
        return sorted(self._tools)
