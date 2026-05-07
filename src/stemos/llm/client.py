from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NUCLEUS_SYSTEM_PROMPT = """
You are Nucleus, a genome mutation proposer. Your job is to propose ONE
specific change to the genome that might improve harness performance.
You do NOT evaluate fitness. You do NOT assign scores. You propose structure changes only.
Output format: JSON with keys {mutation_type, target_field, rationale, new_value}.
""".strip()


GUARDIAN_SYSTEM_PROMPT = """
You are Guardian evaluator. Assess harness output quality using only the fixed criteria below.
Do not suggest genome changes. Score each output independently.
Output format: JSON with keys {criterion_scores: {}, overall: float, reasoning: str}.
""".strip()


def build_guardian_system_prompt(evaluation_criteria: list[dict[str, Any]]) -> str:
    criteria_json = json.dumps(evaluation_criteria, sort_keys=True)
    return f"{GUARDIAN_SYSTEM_PROMPT}\nFixed criteria: {criteria_json}"


class LLMCallLogger:
    def __init__(self, run_dir: str | Path | None = None):
        self.run_dir = Path(run_dir) if run_dir else None

    def configure(self, run_dir: str | Path | None) -> None:
        self.run_dir = Path(run_dir) if run_dir else None

    def record(
        self,
        *,
        role: str,
        prompt: str,
        response: str,
        tokens_used: int | None = None,
    ) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "prompt_hash": self._hash(prompt),
            "response_hash": self._hash(response),
            "tokens_used": int(tokens_used if tokens_used is not None else self._estimate_tokens(prompt, response)),
            "prompt": prompt,
        }
        with (self.run_dir / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        return max(1, (len(prompt) + len(response)) // 4)
