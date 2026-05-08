from stemos.nucleus.scenario_interpreter import ScenarioInterpreter
from stemos.nucleus.model_client import ModelClient
from stemos.scenarios.loader import load_scenario


def test_scenario_loader_reads_yaml_and_jsonl_cases():
    bundle = load_scenario("scenarios/toy_structured_answer")

    assert bundle.scenario.name == "toy_structured_answer"
    assert len(bundle.train_cases) == 2
    assert len(bundle.validation_cases) == 1
    assert bundle.train_cases[0].input["user_request"]


def test_scenario_interpreter_produces_task_diagnosis():
    bundle = load_scenario("scenarios/tiny_task_operator")
    diagnosis = ScenarioInterpreter(ModelClient(test_mode=True)).interpret(bundle.scenario)

    assert diagnosis.task_type == "tiny task operation harness"
    assert diagnosis.expected_task_solving_pattern
    assert diagnosis.likely_failure_modes
    assert diagnosis.likely_needed_capabilities
    assert diagnosis.initial_architecture_hypothesis
    assert diagnosis.initial_evaluation_hypothesis
