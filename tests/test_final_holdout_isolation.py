import json

import yaml

from stem_agent.scenarios.loader import load_scenario


def test_scenario_bundle_loads_external_and_final_holdout(tmp_path):
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(yaml.safe_dump({
        "scenario": {"name": "demo", "description": "d", "task_class": "general"},
        "input_format": {"type": "text", "fields": [{"name": "user_request", "required": True}]},
        "expected_output": {"type": "markdown", "requirements": []},
    }), encoding="utf-8")
    for split in ["train_cases.jsonl", "validation_cases.jsonl"]:
        (scenario_dir / split).write_text(json.dumps({"id": "x", "input": {"user_request": "u"}}) + "\n", encoding="utf-8")
    bundle = load_scenario(scenario_dir)
    assert hasattr(bundle, "external_benchmark_cases")
    assert hasattr(bundle, "final_holdout_cases")
