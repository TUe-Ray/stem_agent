from __future__ import annotations

from pathlib import Path
from typing import Any

from stem_agent.tools.base import Tool


def _call_model(input_data: dict[str, Any]) -> dict[str, Any]:
    return {"text": input_data.get("prompt", "")}


def _read_file(input_data: dict[str, Any]) -> dict[str, Any]:
    path = Path(input_data["path"])
    offset = input_data.get("offset", 1)
    limit = input_data.get("limit")

    content = path.read_text(encoding="utf-8")
    total_lines = content.count("\n") + 1
    total_chars = len(content)

    if limit is not None:
        lines = content.split("\n")
        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        content = "\n".join(lines[start:end])
        return {
            "text": content,
            "total_lines": total_lines,
            "total_chars": total_chars,
            "shown_lines": f"{start+1}-{end}",
        }

    return {
        "text": content,
        "total_lines": total_lines,
        "total_chars": total_chars,
        "note": "File is large. Use offset/limit to read specific sections." if total_lines > 200 else None,
    }


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


def _git_diff(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run git diff HEAD in the workspace to generate an exact patch.

    After modifying source files with write_file, call this to get the
    precise unified diff (100% guaranteed to apply because git produced it).
    This bypasses LLM hallucination of line numbers and context lines.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return {
                "diff": "",
                "error": f"git diff HEAD failed (exit {result.returncode}): {result.stderr.strip()[:500]}",
            }
        diff_text = result.stdout
        if not diff_text.strip():
            return {"diff": "", "note": "No changes detected — workspace is clean."}
        return {"diff": diff_text, "note": f"Generated via git diff HEAD ({len(diff_text)} chars)"}
    except FileNotFoundError:
        return {"diff": "", "error": "git is not installed"}
    except subprocess.TimeoutExpired:
        return {"diff": "", "error": "git diff timed out"}


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
            description="Read a file from the workspace. Use offset/limit for large files. Returns content + total_lines + total_chars.",
            run=_read_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file (relative to workspace root)"},
                    "offset": {"type": "integer", "description": "Start line number (1-indexed, default: 1)"},
                    "limit": {"type": "integer", "description": "Max lines to read (omit to read entire file — DANGEROUS for large files)"},
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
        "git_diff": Tool(
            name="git_diff",
            description="Run git diff HEAD to generate an EXACT unified diff from file modifications. Use this AFTER modifying source files with write_file — the diff is 100% guaranteed correct (git generates it, not the LLM).",
            run=_git_diff,
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
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
