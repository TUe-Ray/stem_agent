import json

import yaml

from stem_agent.config import Settings
from stem_agent.evolution.loop import EvolutionLoop
from stem_agent.kernel.signal_audit import SignalLeakAuditor, nucleus_visible_artifact_paths
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


def test_external_benchmark_and_final_holdout_artifacts_are_isolated(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )

    result = loop.evolve("scenarios/toy_structured_answer", "holdout_isolation")
    run_dir = result.run_dir

    assert (run_dir / "generation_000" / "external_benchmark" / "eval_result.json").exists()
    assert (run_dir / "final_evaluation" / "external_benchmark" / "eval_result.json").exists()
    assert (run_dir / "final_holdout" / "eval_result.json").exists()
    report = result.report_path.read_text(encoding="utf-8")
    assert "## Research Evaluation" in report
    assert "Final holdout" in report

    bundle = load_scenario("scenarios/toy_structured_answer")
    ids = [case.id for case in bundle.final_holdout_cases + bundle.external_benchmark_cases]
    texts = [str(case.input) for case in bundle.final_holdout_cases + bundle.external_benchmark_cases]
    auditor = SignalLeakAuditor(forbidden_case_ids=ids, forbidden_case_texts=texts)
    for path in nucleus_visible_artifact_paths(run_dir):
        auditor.assert_no_leak_in_file(path)
