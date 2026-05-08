from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from stemos.scenarios.schema import Scenario, TaskCase


class ScenarioBundle(BaseModel):
    path: Path
    scenario: Scenario
    train_cases: list[TaskCase]
    validation_cases: list[TaskCase]


def _read_jsonl(path: Path) -> list[TaskCase]:
    if not path.exists():
        return []
    cases: list[TaskCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(TaskCase.model_validate(json.loads(line)))
    return cases


def load_scenario(path: str | Path) -> ScenarioBundle:
    root = Path(path)
    if root.is_file():
        root = root.parent
    scenario_path = root / "scenario.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario.yaml at {scenario_path}")

    data = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    scenario = Scenario.model_validate(data)
    scenario.convergence_policy_explicit = "convergence_policy" in data
    if not scenario.convergence_policy_explicit:
        scenario.convergence_policy = scenario.convergence_policy.model_copy(
            update={
                "max_generations": scenario.evolution.max_generations,
                "absolute_score_threshold": 1.01,
            }
        )
    return ScenarioBundle(
        path=root,
        scenario=scenario,
        train_cases=_read_jsonl(root / "train_cases.jsonl"),
        validation_cases=_read_jsonl(root / "validation_cases.jsonl"),
    )
