"""Pre-built tool catalog for stem_agent.

Each catalog tool is hand-written, tested, and self-describing.
Nucleus can discover tools via the catalog and propose add_tool_to_genome
mutations that reference existing tools instead of generating code from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CatalogTool:
    """A pre-built, discoverable tool for the evolution harness."""

    name: str
    description: str           # 1-2 sentence description for Nucleus prompt
    category: str              # web, data, code, text, safety, reasoning, files
    parameter_schema: dict     # JSON Schema for parameters
    examples: list[dict]       # 2-3 input→output examples for Nucleus
    cost_estimate: float = 0.0  # Estimated API cost or seconds per call
    fn: Callable | None = None  # The actual Python function


class ToolCatalog:
    """Registry of all pre-built tools, organized by category."""

    def __init__(self):
        self._tools: dict[str, CatalogTool] = {}

    def register(self, tool: CatalogTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> CatalogTool | None:
        return self._tools.get(name)

    def list_all(self) -> list[CatalogTool]:
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[CatalogTool]:
        return [t for t in self._tools.values() if t.category == category]

    def categories(self) -> list[str]:
        return sorted(set(t.category for t in self._tools.values()))

    def render_for_nucleus(self, max_per_category: int = 5) -> str:
        """Render a compact tool catalog block for the Nucleus prompt."""
        lines = ["AVAILABLE TOOL CATALOG (use add_tool_to_genome to adopt):", ""]
        for cat in self.categories():
            tools = self.list_by_category(cat)[:max_per_category]
            if not tools:
                continue
            lines.append(f"## {cat}")
            for tool in tools:
                params = ", ".join(
                    f"{k}: {v.get('type', 'any')}" for k, v in tool.parameter_schema.get("properties", {}).items()
                )
                lines.append(f"- **{tool.name}**: {tool.description}")
                if params:
                    lines.append(f"  Params: {{{params}}}")
                if tool.examples:
                    example = tool.examples[0]
                    lines.append(f"  Example: {tool.name}({example.get('input', '...')}) → ...")
            lines.append("")
        return "\n".join(lines)


# Singleton catalog
_catalog: ToolCatalog | None = None


def get_catalog() -> ToolCatalog:
    global _catalog
    if _catalog is None:
        _catalog = ToolCatalog()
        _register_all(_catalog)
    return _catalog


def _register_all(catalog: ToolCatalog) -> None:
    """Register all built-in catalog tools. Called once on first access."""
    from stem_agent.tools.catalog import data, code, text, safety, reasoning, files
    for module in [data, code, text, safety, reasoning, files]:
        if hasattr(module, "register"):
            module.register(catalog)
