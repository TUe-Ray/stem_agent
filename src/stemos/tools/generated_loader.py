from __future__ import annotations

import importlib.util
from pathlib import Path


def load_generated_tool(path: str | Path):
    tool_path = Path(path)
    spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generated tool at {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"Generated tool {tool_path} must expose run(input_data)")
    return module.run
