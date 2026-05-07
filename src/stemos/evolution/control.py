from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from stemos.genome.models import Genome
from stemos.genome.serializer import save_genome
from stemos.kernel.evaluator import EvaluationResult


CONTROL_COMMANDS = {"pause", "resume", "freeze_now", "abort"}


class ControlState(BaseModel):
    run_id: str
    command: str | None = None
    status: str = "IDLE"
    freeze_best: bool = False
    message: str = ""


class CheckpointState(BaseModel):
    run_id: str
    scenario_hash: str
    generation: int
    next_generation: int
    phase: str
    current_genome_path: str
    best_genome_path: str
    current_parent_id: str | None = None
    baseline_eval: dict[str, Any] | None = None
    best_eval: dict[str, Any] | None = None
    archive: dict[str, Any] | None = None
    stagnation_count: int = 0
    patience_left: int
    stop_reason: str = ""


class EvolutionControl:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = run_dir
        self.run_id = run_id
        self.control_path = run_dir / "control.json"
        self.checkpoint_dir = run_dir / "checkpoint" / "latest"

    def write_command(
        self, command: str, *, freeze_best: bool = False, status: str = "REQUESTED"
    ) -> ControlState:
        if command not in CONTROL_COMMANDS:
            raise ValueError(f"Unsupported control command: {command}")
        state = ControlState(
            run_id=self.run_id,
            command=command,
            status=status,
            freeze_best=freeze_best,
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.control_path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return state

    def set_status(
        self,
        status: str,
        *,
        command: str | None = None,
        message: str = "",
        freeze_best: bool = False,
    ) -> ControlState:
        state = ControlState(
            run_id=self.run_id,
            command=command,
            status=status,
            message=message,
            freeze_best=freeze_best,
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.control_path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return state

    def read(self) -> ControlState:
        if not self.control_path.exists():
            return ControlState(run_id=self.run_id, status="RUNNING")
        return ControlState.model_validate(
            json.loads(self.control_path.read_text(encoding="utf-8"))
        )

    def clear_command(self, status: str = "RUNNING") -> None:
        current = self.read()
        current.command = None
        current.status = status
        self.control_path.write_text(
            json.dumps(current.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def save_checkpoint(
        self,
        *,
        scenario_hash: str,
        generation: int,
        next_generation: int,
        phase: str,
        current_genome: Genome,
        best_genome: Genome,
        baseline_eval: EvaluationResult | None = None,
        best_eval: EvaluationResult | None = None,
        archive: dict[str, Any] | None = None,
        stagnation_count: int = 0,
        patience_left: int = 0,
        stop_reason: str = "",
        current_parent_id: str | None = None,
    ) -> CheckpointState:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        current_path = self.checkpoint_dir / "current_genome.yaml"
        best_path = self.checkpoint_dir / "best_genome.yaml"
        save_genome(current_genome, current_path)
        save_genome(best_genome, best_path)
        state = CheckpointState(
            run_id=self.run_id,
            scenario_hash=scenario_hash,
            generation=generation,
            next_generation=next_generation,
            phase=phase,
            current_genome_path=str(current_path),
            best_genome_path=str(best_path),
            current_parent_id=current_parent_id,
            baseline_eval=baseline_eval.model_dump(mode="json") if baseline_eval else None,
            best_eval=best_eval.model_dump(mode="json") if best_eval else None,
            archive=archive,
            stagnation_count=stagnation_count,
            patience_left=patience_left,
            stop_reason=stop_reason,
        )
        (self.checkpoint_dir / "state.json").write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return state

    def load_checkpoint(self) -> CheckpointState:
        state_path = self.checkpoint_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing checkpoint state: {state_path}")
        return CheckpointState.model_validate(
            json.loads(state_path.read_text(encoding="utf-8"))
        )


def scenario_hash_from_file(path: Path) -> str:
    scenario_file = path / "scenario.yaml" if path.is_dir() else path
    data = scenario_file.read_bytes()
    return hashlib.sha256(data).hexdigest()
