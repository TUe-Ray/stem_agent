from __future__ import annotations

from pathlib import Path
from typing import Any

from stemos.tools.base import Tool


def _call_model(input_data: dict[str, Any]) -> dict[str, Any]:
    return {"text": input_data.get("prompt", "")}


def _read_file(input_data: dict[str, Any]) -> dict[str, Any]:
    path = Path(input_data["path"])
    return {"text": path.read_text(encoding="utf-8")}


def _write_file(input_data: dict[str, Any]) -> dict[str, Any]:
    path = Path(input_data["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(input_data.get("text", ""), encoding="utf-8")
    return {"path": str(path)}


def _run_python(input_data: dict[str, Any]) -> dict[str, Any]:
    # The MVP exposes this as a registered capability, but generated code is
    # still governed by Guardian sandbox checks before activation.
    code = input_data.get("code", "")
    namespace: dict[str, Any] = {}
    exec(code, {"__builtins__": {"len": len, "range": range, "min": min, "max": max}}, namespace)
    return {"namespace": namespace}


def _inspect_workspace(input_data: dict[str, Any]) -> dict[str, Any]:
    root = Path(input_data.get("path", "."))
    return {"files": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())}


def builtin_tools() -> dict[str, Tool]:
    return {
        "call_model": Tool(name="call_model", description="Call the configured model.", run=_call_model),
        "read_file": Tool(name="read_file", description="Read a file.", run=_read_file),
        "write_file": Tool(name="write_file", description="Write a file.", run=_write_file),
        "run_python": Tool(name="run_python", description="Run bounded Python code.", run=_run_python),
        "inspect_workspace": Tool(
            name="inspect_workspace",
            description="Inspect workspace files.",
            run=_inspect_workspace,
        ),
    }
