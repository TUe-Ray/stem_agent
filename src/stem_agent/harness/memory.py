from __future__ import annotations

from typing import Any


class MemoryStore:
    def __init__(self, memory_spec: dict[str, Any]):
        self.memory_spec = memory_spec
        self.values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def snapshot(self) -> dict[str, Any]:
        return dict(self.values)
