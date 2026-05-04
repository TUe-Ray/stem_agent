from __future__ import annotations

from pathlib import Path

import yaml

from stemos.genome.models import Genome


def _package_default_path() -> Path:
    return Path(__file__).with_name("default_genome.yaml")


def load_genome(path: str | Path) -> Genome:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Genome.model_validate(data)


def load_default_genome() -> Genome:
    return load_genome(_package_default_path())
