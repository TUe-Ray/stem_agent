import json
from pathlib import Path

from stem_agent.benchmarks.robustness_suite import ROBUSTNESS_BENCHMARKS
from stem_agent.scenarios.loader import load_scenario


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_robustness_benchmark_manifest_covers_distinct_task_classes():
    assert {item.task_class for item in ROBUSTNESS_BENCHMARKS} == {
        "security_review_triage",
        "deep_research_synthesis",
        "release_quality_triage",
    }
    assert all(item.name.endswith("_demo") for item in ROBUSTNESS_BENCHMARKS)
    assert all(item.path.endswith("_demo") for item in ROBUSTNESS_BENCHMARKS)


def test_robustness_benchmarks_load_with_generalization_splits():
    for benchmark in ROBUSTNESS_BENCHMARKS:
        bundle = load_scenario(benchmark.path)

        assert bundle.scenario.name == benchmark.name
        assert bundle.scenario.scenario.task_class == benchmark.task_class
        assert len(bundle.train_cases) >= 2
        assert len(bundle.validation_cases) >= 2
        assert len(bundle.external_benchmark_cases) >= 2
        assert len(bundle.final_holdout_cases) >= 2
        assert bundle.scenario.evolution.signal_mode == "internal_plus_external_score"
        assert 0.0 < bundle.scenario.evolution.external_signal_weight <= 0.30
        assert bundle.scenario.domain_tags
        assert "call_model" in bundle.scenario.available_builtin_tools


def test_robustness_benchmark_splits_do_not_reuse_inputs():
    for benchmark in ROBUSTNESS_BENCHMARKS:
        bundle = load_scenario(benchmark.path)
        split_cases = {
            "train": [case.model_dump(mode="json") for case in bundle.train_cases],
            "validation": [case.model_dump(mode="json") for case in bundle.validation_cases],
            "external": [case.model_dump(mode="json") for case in bundle.external_benchmark_cases],
            "hidden": _read_jsonl(Path(benchmark.path) / "hidden_cases.jsonl"),
            "final": [case.model_dump(mode="json") for case in bundle.final_holdout_cases],
        }
        seen_inputs: dict[str, str] = {}
        seen_ids: dict[str, str] = {}

        for split, cases in split_cases.items():
            assert cases, f"{benchmark.name} missing {split} cases"
            for case in cases:
                case_id = str(case["id"])
                input_signature = json.dumps(case["input"], sort_keys=True)
                assert case_id not in seen_ids, (
                    f"{benchmark.name} reuses case id {case_id} in "
                    f"{split} and {seen_ids.get(case_id)}"
                )
                assert input_signature not in seen_inputs, (
                    f"{benchmark.name} reuses case input in "
                    f"{split} and {seen_inputs.get(input_signature)}"
                )
                seen_ids[case_id] = split
                seen_inputs[input_signature] = split
