from stem_agent.nucleus.scenario_interpreter import ScenarioInterpreter
from stem_agent.nucleus.model_client import ModelClient
from stem_agent.cli import _gsm8k_benchmark_preset, _gsm8k_scenario_yaml
from stem_agent.scenarios.loader import load_scenario


def test_scenario_loader_reads_yaml_and_jsonl_cases():
    bundle = load_scenario("scenarios/toy_structured_answer")

    assert bundle.scenario.name == "toy_structured_answer"
    assert len(bundle.train_cases) == 2
    assert len(bundle.validation_cases) == 2
    assert len(bundle.external_benchmark_cases) == 2
    assert len(bundle.final_holdout_cases) == 2
    assert bundle.train_cases[0].input["user_request"]


def test_gsm8k_demo_scenario_loads_as_smoke_benchmark():
    bundle = load_scenario("scenarios/gsm8k_demo")

    assert bundle.scenario.name == "gsm8k_demo"
    assert bundle.train_cases
    assert bundle.validation_cases
    assert "final numeric answer" in " ".join(bundle.scenario.expected_output.requirements)


def test_scenario_interpreter_produces_task_diagnosis():
    bundle = load_scenario("scenarios/tiny_task_operator")
    diagnosis = ScenarioInterpreter(ModelClient(test_mode=True)).interpret(bundle.scenario)

    assert diagnosis.task_type == "tiny task operation harness"
    assert diagnosis.expected_task_solving_pattern
    assert diagnosis.likely_failure_modes
    assert diagnosis.likely_needed_capabilities
    assert diagnosis.initial_architecture_hypothesis
    assert diagnosis.initial_evaluation_hypothesis


def test_gsm8k_demo_and_full_presets_are_named_clearly():
    demo = _gsm8k_benchmark_preset("gsm8k_demo")
    full = _gsm8k_benchmark_preset("gsm8k_full")
    legacy = _gsm8k_benchmark_preset("gsm8k_mini")

    assert demo["scenario_name"] == "gsm8k_demo"
    assert demo["preset"] == "demo"
    assert full["scenario_name"] == "gsm8k_full"
    assert full["preset"] == "full"
    assert legacy["scenario_name"] == "gsm8k_demo"
    assert full["n_train"] > demo["n_train"]

    demo_scenario = _gsm8k_scenario_yaml("gsm8k_demo", preset="demo")
    full_scenario = _gsm8k_scenario_yaml("gsm8k_full", preset="full")

    assert demo_scenario["scenario"]["name"] == "gsm8k_demo"
    assert full_scenario["scenario"]["name"] == "gsm8k_full"
    assert full_scenario["evolution"]["max_generations"] > demo_scenario["evolution"]["max_generations"]
