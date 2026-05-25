"""File tools: find files, grep, line counts, file statistics."""

import re
from pathlib import Path
from typing import Any

from stem_agent.tools.catalog import CatalogTool


def _find_files(input_data: dict[str, Any]) -> dict[str, Any]:
    """Find files by glob pattern in a directory tree."""
    pattern = input_data.get("pattern", "*")
    root = Path(input_data.get("path", ".")).resolve()
    recursive = input_data.get("recursive", True)

    if recursive:
        matches = sorted(root.rglob(pattern))
    else:
        matches = sorted(root.glob(pattern))

    files_only = [str(m.relative_to(root)) for m in matches if m.is_file()]
    return {
        "pattern": pattern,
        "count": len(files_only),
        "files": files_only[:50],
    }


def _grep(input_data: dict[str, Any]) -> dict[str, Any]:
    """Search file contents for a regex pattern. Returns matches with context."""
    pattern = input_data.get("pattern", "")
    root = Path(input_data.get("path", ".")).resolve()
    file_glob = input_data.get("file_glob", "*")
    max_matches = input_data.get("max_matches", 50)
    context_lines = input_data.get("context_lines", 0)

    if not pattern:
        return {"matches": [], "error": "No pattern provided"}

    matches = []
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"matches": [], "error": f"Invalid regex: {e}"}

    for filepath in sorted(root.rglob(file_glob)):
        if not filepath.is_file():
            continue
        if filepath.suffix not in {".py", ".txt", ".md", ".yaml", ".yml", ".json", ".jsonl",
                                    ".toml", ".cfg", ".ini", ".sh", ".bash", ".c", ".h",
                                    ".cpp", ".rs", ".java", ".js", ".ts", ".html", ".css"}:
            continue
        try:
            lines = filepath.read_text(encoding="utf-8", errors="replace").split("\n")
        except Exception:
            continue

        for i, line in enumerate(lines):
            if compiled.search(line):
                ctx_start = max(0, i - context_lines)
                ctx_end = min(len(lines), i + context_lines + 1)
                context = lines[ctx_start:ctx_end]
                matches.append({
                    "file": str(filepath.relative_to(root)),
                    "line": i + 1,
                    "text": line.strip()[:200],
                    "context": context if context_lines > 0 else None,
                })
                if len(matches) >= max_matches:
                    break
        if len(matches) >= max_matches:
            break

    return {"matches": matches, "count": len(matches), "pattern": pattern}


def _count_lines(input_data: dict[str, Any]) -> dict[str, Any]:
    """Count lines of code in a file or directory."""
    target = Path(input_data.get("path", ".")).resolve()
    file_glob = input_data.get("file_glob", "*.py")

    if target.is_file():
        files = [target]
    else:
        files = list(target.rglob(file_glob))

    total_lines = 0
    total_chars = 0
    file_counts = []
    for f in files:
        if f.is_file():
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                lines = content.count("\n") + 1
                total_lines += lines
                total_chars += len(content)
                file_counts.append({"file": str(f.relative_to(target)), "lines": lines})
            except Exception:
                continue

    return {
        "file_count": len(file_counts),
        "total_lines": total_lines,
        "total_chars": total_chars,
        "files": file_counts[:30],
    }


def _file_stats(input_data: dict[str, Any]) -> dict[str, Any]:
    """Get file metadata: size, modified time, permissions."""
    path = Path(input_data.get("path", "."))
    if not path.exists():
        return {"error": f"Path not found: {path}"}

    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_human": _human_size(stat.st_size),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "modified": stat.st_mtime,
    }


def _human_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def register(catalog):
    catalog.register(CatalogTool(
        name="find_files",
        description="Find files by glob pattern in a directory tree. Returns relative paths.",
        category="files",
        parameter_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. *.py"},
                "path": {"type": "string", "description": "Root directory"},
                "recursive": {"type": "boolean", "description": "Search subdirectories"},
            },
            "required": ["pattern"],
        },
        examples=[
            {"input": 'pattern="*.py" path="src/" recursive=true', "output": "{count: 42, files: [...]}"},
        ],
        cost_estimate=0.1,
        fn=_find_files,
    ))
    catalog.register(CatalogTool(
        name="grep",
        description="Search file contents for a regex pattern. Returns file, line number, matched text.",
        category="files",
        parameter_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search in"},
                "file_glob": {"type": "string", "description": "File filter, e.g. *.py"},
                "max_matches": {"type": "integer", "description": "Max results", "default": 50},
                "context_lines": {"type": "integer", "description": "Lines before/after match", "default": 0},
            },
            "required": ["pattern"],
        },
        examples=[
            {"input": 'pattern="TODO" path="src/" file_glob="*.py"', "output": "{matches: [...], count: 5}"},
        ],
        cost_estimate=0.5,
        fn=_grep,
    ))
    catalog.register(CatalogTool(
        name="count_lines",
        description="Count lines of code in a file or directory tree. Returns total lines + per-file.",
        category="files",
        parameter_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path"},
                "file_glob": {"type": "string", "description": "File filter", "default": "*.py"},
            },
            "required": [],
        },
        examples=[
            {"input": 'path="src/" file_glob="*.py"', "output": "{file_count:15, total_lines: 3200}"},
        ],
        cost_estimate=0.2,
        fn=_count_lines,
    ))
    catalog.register(CatalogTool(
        name="file_stats",
        description="Get file size, modified time, permissions for a path.",
        category="files",
        parameter_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path"},
            },
            "required": ["path"],
        },
        examples=[
            {"input": 'path="README.md"', "output": "{size_human: '4.2 KB', is_file: true}"},
        ],
        cost_estimate=0.0,
        fn=_file_stats,
    ))
