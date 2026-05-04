from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from stemos.kernel.validators import ValidationResult


FORBIDDEN_IMPORTS = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "shutil",
    "pathlib",
}


class Sandbox:
    """Static and runtime checks for generated tools."""

    def validate_generated_tool_code(self, code: str) -> ValidationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult.reject(f"Syntax error in generated tool: {exc}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0]
                    if root_name in FORBIDDEN_IMPORTS:
                        return ValidationResult.reject(f"Forbidden import: {root_name}")
            if isinstance(node, ast.ImportFrom):
                root_name = (node.module or "").split(".")[0]
                if root_name in FORBIDDEN_IMPORTS:
                    return ValidationResult.reject(f"Forbidden import: {root_name}")
        return ValidationResult.allow("generated tool passed static import checks")

    def validate_shell_command(self, command_spec: dict) -> ValidationResult:
        if command_spec.get("command") and not command_spec.get("timeout_seconds"):
            return ValidationResult.reject("Shell command requires timeout_seconds")
        return ValidationResult.allow("shell command spec is bounded")

    def run_generated_tool_tests(self, tool_dir: Path, timeout_seconds: int = 10) -> ValidationResult:
        if not tool_dir.exists():
            return ValidationResult.reject(f"Generated tool directory does not exist: {tool_dir}")

        command = [
            sys.executable,
            "-m",
            "pytest",
            str(tool_dir),
            "-q",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=tool_dir.parent,
                timeout=timeout_seconds,
                text=True,
                capture_output=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult.reject("Generated tool tests timed out")

        if completed.returncode != 0:
            return ValidationResult.reject(
                "Generated tool tests failed: "
                + (completed.stdout + completed.stderr).strip()
            )
        return ValidationResult.allow("generated tool tests passed")
