from __future__ import annotations

from pathlib import Path

import yaml

from stem_agent.genome.models import Genome


def genome_to_dict(genome: Genome) -> dict:
    return genome.model_dump(mode="json", exclude_none=False)


def save_genome(genome: Genome, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(genome_to_dict(genome), sort_keys=False),
        encoding="utf-8",
    )
