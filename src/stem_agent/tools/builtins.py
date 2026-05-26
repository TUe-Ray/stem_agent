from __future__ import annotations

from pathlib import Path
from typing import Any

from stem_agent.tools.base import Tool


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


def _search_code(input_data: dict[str, Any]) -> dict[str, Any]:
    """Search workspace files for regex matches. Input: {pattern, path?}."""
    import re
    root = Path(input_data.get("path", "."))
    pattern = input_data.get("pattern", "")
    if not pattern:
        return {"matches": []}
    matches: list[dict[str, Any]] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix not in {".py", ".c", ".h", ".rs", ".java", ".js", ".ts"}:
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for lineno, line in enumerate(content.split("\n"), 1):
            if re.search(pattern, line):
                matches.append({"file": str(file_path.relative_to(root)), "line": lineno, "text": line.strip()[:200]})
                if len(matches) >= 50:
                    break
        if len(matches) >= 50:
            break
    return {"matches": matches, "pattern": pattern}


def _write_patch(input_data: dict[str, Any]) -> dict[str, Any]:
    """Write a unified-diff patch to workspace/predicted.patch."""
    patch_dir = Path("artifacts")
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / "predicted.patch"
    patch_path.write_text(input_data.get("patch_text", ""), encoding="utf-8")
    return {"path": str(patch_path), "size": patch_path.stat().st_size}


def _apply_patch_dry_run(input_data: dict[str, Any]) -> dict[str, Any]:
    """Check if a unified diff applies cleanly via patch --dry-run."""
    import subprocess
    import tempfile
    patch_text = input_data.get("patch_text", "")
    if not patch_text.strip():
        return {"applies": False, "error": "Empty patch"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp:
        tmp.write(patch_text)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["patch", "--dry-run", "-p1", "-f", "-i", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        applies = result.returncode == 0
        return {"applies": applies, "error": result.stderr.strip()[:500] if not applies else ""}
    except subprocess.TimeoutExpired:
        return {"applies": False, "error": "patch --dry-run timed out"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def builtin_tools() -> dict[str, Tool]:
    return {
        "call_model": Tool(name="call_model", description="Call the configured model.", run=_call_model),
        "read_file": Tool(
            name="read_file",
            description="Read a file from the workspace. Returns the file contents as text.",
            run=_read_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read (relative to workspace root)"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        "write_file": Tool(
            name="write_file",
            description="Write text content to a file in the workspace. Creates parent directories.",
            run=_write_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write the file to"},
                    "text": {"type": "string", "description": "Text content to write"},
                },
                "required": ["path", "text"],
                "additionalProperties": False,
            },
        ),
        "run_python": Tool(
            name="run_python",
            description="Run bounded Python code. Input: {code}. Returns namespace dict.",
            run=_run_python,
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        ),
        "inspect_workspace": Tool(
            name="inspect_workspace",
            description="List all files in the workspace directory.",
            run=_inspect_workspace,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Root directory to inspect (default: '.')"},
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        "search_code": Tool(
            name="search_code",
            description="Search source code files for a regex pattern. Returns file, line number, and matched text.",
            run=_search_code,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for in source files"},
                    "path": {"type": "string", "description": "Directory to search in (default: '.')"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        ),
        "write_patch": Tool(
            name="write_patch",
            description="Write a unified-diff patch to artifacts/predicted.patch. Call this when you have a complete fix.",
            run=_write_patch,
            parameters={
                "type": "object",
                "properties": {
                    "patch_text": {"type": "string", "description": "The complete unified diff patch text, starting with '--- a/' or 'diff --git'"},
                },
                "required": ["patch_text"],
                "additionalProperties": False,
            },
        ),
        "apply_patch_dry_run": Tool(
            name="apply_patch_dry_run",
            description="Test if a unified diff applies cleanly before writing it. Use this to verify your patch.",
            run=_apply_patch_dry_run,
            parameters={
                "type": "object",
                "properties": {
                    "patch_text": {"type": "string", "description": "The unified diff to test"},
                },
                "required": ["patch_text"],
                "additionalProperties": False,
            },
        ),
    }
