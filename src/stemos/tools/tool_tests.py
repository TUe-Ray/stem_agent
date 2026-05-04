from __future__ import annotations

from pathlib import Path


def has_test_for_tool(tool_name: str, test_path: str | None) -> bool:
    if not test_path:
        return False
    path = Path(test_path)
    return path.exists() and tool_name in path.read_text(encoding="utf-8")
