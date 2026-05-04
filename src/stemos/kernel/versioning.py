from __future__ import annotations

import shutil
from pathlib import Path

from stemos.genome.models import Genome
from stemos.genome.serializer import save_genome


class VersionStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.snapshots_dir = run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_genome(self, version_id: str, genome: Genome) -> Path:
        path = self.snapshots_dir / f"{version_id}.yaml"
        save_genome(genome, path)
        return path

    def rollback_artifacts(self, version_id: str, target_path: Path) -> None:
        source = self.snapshots_dir / f"{version_id}.yaml"
        if not source.exists():
            raise FileNotFoundError(f"Missing snapshot: {source}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_path)
