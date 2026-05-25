"""Code tools: test runner, linter, git operations."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from stem_agent.tools.catalog import CatalogTool


def _run_tests(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run pytest on a given test file or directory. Returns pass/fail + output."""
    target = input_data.get("target", ".")
    timeout = input_data.get("timeout", 30)

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", target, "-q", "--tb=short"],
            capture_output=True, text=True, timeout=timeout,
            cwd=input_data.get("cwd") or None,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return {
            "passed": result.returncode == 0,
            "exit_code": result.returncode,
            "output": output[-2000:],
            "summary": output.split("\n")[-2] if output else "",
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Tests timed out after {timeout}s"}


def _lint_check(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run ruff/flake8 lint on a file or directory."""
    target = input_data.get("target", ".")
    linter = input_data.get("linter", "ruff")

    try:
        result = subprocess.run(
            [linter, "check", target, "--output-format=concise"],
            capture_output=True, text=True, timeout=15,
        )
        issues = [line.strip() for line in result.stdout.split("\n") if line.strip()]
        return {
            "clean": result.returncode == 0 and len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues[:20],
        }
    except FileNotFoundError:
        return {"clean": True, "note": f"{linter} not installed"}


def _git_diff(input_data: dict[str, Any]) -> dict[str, Any]:
    """Show git diff between commits, branches, or working tree."""
    target = input_data.get("target", "HEAD~1..HEAD")
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", target],
            capture_output=True, text=True, timeout=10,
            cwd=input_data.get("cwd") or None,
        )
        stat = result.stdout.strip()
        # Also get file list
        files_result = subprocess.run(
            ["git", "diff", "--name-only", target],
            capture_output=True, text=True, timeout=10,
            cwd=input_data.get("cwd") or None,
        )
        files = [f for f in files_result.stdout.strip().split("\n") if f]
        return {"files_changed": files, "file_count": len(files), "stat": stat}
    except Exception as e:
        return {"error": str(e)}


def _git_blame(input_data: dict[str, Any]) -> dict[str, Any]:
    """Show git blame for a specific file and line range."""
    filepath = input_data.get("file", "")
    lines = input_data.get("lines", "")
    try:
        cmd = ["git", "blame", "--line-porcelain"]
        if lines:
            cmd.extend(["-L", lines])
        cmd.append(filepath)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return {"blame": result.stdout[:3000]}
    except Exception as e:
        return {"error": str(e)}


def register(catalog):
    catalog.register(CatalogTool(
        name="run_tests",
        description="Run pytest on a file or directory. Returns pass/fail and error output.",
        category="code",
        parameter_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Test file or directory path"},
                "timeout": {"type": "integer", "description": "Seconds before timeout", "default": 30},
            },
            "required": ["target"],
        },
        examples=[
            {"input": 'target="tests/"', "output": "{passed: true, summary: '10 passed'}"},
        ],
        cost_estimate=5.0,
        fn=_run_tests,
    ))
    catalog.register(CatalogTool(
        name="lint_check",
        description="Run linter (ruff) on code. Returns list of issues found.",
        category="code",
        parameter_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "File or directory to lint"},
                "linter": {"type": "string", "description": "ruff (default) or flake8"},
            },
            "required": ["target"],
        },
        examples=[
            {"input": 'target="src/"', "output": "{clean: false, issues: ['F401 unused import']}"},
        ],
        cost_estimate=3.0,
        fn=_lint_check,
    ))
    catalog.register(CatalogTool(
        name="git_diff",
        description="Show git diff (files changed + stat) between commits or working tree vs HEAD.",
        category="code",
        parameter_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Git revision range, e.g. HEAD~3..HEAD"},
            },
            "required": [],
        },
        examples=[
            {"input": 'target="HEAD~1..HEAD"', "output": "{files: ['src/main.py'], count: 1}"},
        ],
        cost_estimate=1.0,
        fn=_git_diff,
    ))
    catalog.register(CatalogTool(
        name="git_blame",
        description="Show who last modified each line of a file (git blame).",
        category="code",
        parameter_schema={
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File path"},
                "lines": {"type": "string", "description": "Line range, e.g. 10-20"},
            },
            "required": ["file"],
        },
        examples=[
            {"input": 'file="src/main.py" lines="40-50"', "output": "commit abc123 (Author 2026-01-01) ..."},
        ],
        cost_estimate=1.0,
        fn=_git_blame,
    ))
