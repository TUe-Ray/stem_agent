from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from stem_agent.kernel.validators import ValidationResult


FORBIDDEN_IMPORTS = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "shutil",
}

# pathlib is allowed, but specific write/destructive methods are blocked at AST level
FORBIDDEN_PATHLIB_METHODS = {
    "write_text", "write_bytes", "unlink", "rmdir", "mkdir",
    "rename", "replace", "touch", "chmod", "symlink_to", "hardlink_to",
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
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_PATHLIB_METHODS:
                if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') == 'Path':
                    return ValidationResult.reject(
                        f"Forbidden pathlib method on Path object: {node.attr}"
                    )
                if isinstance(node.value, ast.Name) and self._is_pathlib_path(node.value, tree):
                    return ValidationResult.reject(
                        f"Forbidden pathlib method on Path variable: {node.attr}"
                    )
        return ValidationResult.allow("generated tool passed static import checks")

    def _is_pathlib_path(self, name_node: ast.Name, tree: ast.AST) -> bool:
        """Heuristic: check if a variable was assigned from Path()."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name_node.id:
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'Path':
                                return True
        return False

    def validate_shell_command(self, command_spec: dict) -> ValidationResult:
        if command_spec.get("command") and not command_spec.get("timeout_seconds"):
            return ValidationResult.reject("Shell command requires timeout_seconds")
        return ValidationResult.allow("shell command spec is bounded")

    def run_generated_tool_tests(self, tool_dir: Path, timeout_seconds: int = 10) -> ValidationResult:
        tool_dir = tool_dir.resolve()
        if not tool_dir.exists():
            return ValidationResult.reject(f"Generated tool directory does not exist: {tool_dir}")

        command = [
            sys.executable,
            "-m",
            "pytest",
            ".",
            "-q",
            "--capture=no",
            "-p",
            "no:cacheprovider",
        ]
        try:
            env = os.environ.copy()
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            completed = subprocess.run(
                command,
                cwd=tool_dir,
                timeout=timeout_seconds,
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult.reject("Generated tool tests timed out")

        if completed.returncode != 0:
            return ValidationResult.reject(
                "Generated tool tests failed: "
                + (completed.stdout + completed.stderr).strip()
            )
        return ValidationResult.allow("generated tool tests passed")
