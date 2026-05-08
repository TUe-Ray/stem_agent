from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stemos.genome.models import Genome
from stemos.genome.models import WorkflowStep


class IntraTestAdapter:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self._records: list[dict[str, Any]] = []

    def active(self, genome: Genome) -> bool:
        return bool(genome.intra_test_reflection_enabled)

    def max_retries(self, genome: Genome) -> int:
        return max(int(genome.max_intra_reflection_retries), 0)

    def reflection_for_failure(
        self,
        *,
        step: WorkflowStep,
        gate_trace: dict[str, Any],
    ) -> str:
        gate_name = str(gate_trace.get("gate") or "quality_gate")
        missing = gate_trace.get("missing") or []
        missing_text = ", ".join(str(item) for item in missing) if missing else "the gate requirements"
        return (
            f"Retry step {step.id} with a concise correction: satisfy {gate_name} "
            f"by covering {missing_text}."
        )

    def record(
        self,
        *,
        task_id: str,
        step_name: str,
        quality_gate_name: str,
        attempt_number: int,
        reflection_text: str,
        retry_improved_output: bool,
    ) -> None:
        self._records.append(
            {
                "task_id": task_id,
                "step_name": step_name,
                "quality_gate_name": quality_gate_name,
                "attempt_number": attempt_number,
                "reflection_text": reflection_text,
                "retry_improved_output": retry_improved_output,
            }
        )

    def flush(self) -> None:
        if not self._records:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "intra_adaptation_log.jsonl").open("a", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._records.clear()
