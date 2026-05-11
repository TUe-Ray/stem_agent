from stem_agent.scenarios.loader import load_scenario


def test_demo_scenario_loads():
    bundle = load_scenario("scenarios/stem_demo_structured_qa")

    assert bundle.scenario.name == "stem_demo_structured_qa"
    assert bundle.train_cases
    assert bundle.validation_cases
    assert bundle.scenario.expected_output.requirements
