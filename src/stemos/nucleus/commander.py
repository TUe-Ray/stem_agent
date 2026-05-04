from __future__ import annotations

from stemos.genome.models import Genome
from stemos.nucleus.failure_analyzer import FailureAnalyzer
from stemos.nucleus.mutation_planner import MutationPlanner
from stemos.nucleus.scenario_interpreter import ScenarioInterpreter
from stemos.nucleus.schemas import MutationPlan, TaskDiagnosis
from stemos.scenarios.schema import Scenario


class NucleusCommander:
    def __init__(
        self,
        scenario_interpreter: ScenarioInterpreter | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        mutation_planner: MutationPlanner | None = None,
    ):
        self.scenario_interpreter = scenario_interpreter or ScenarioInterpreter()
        self.failure_analyzer = failure_analyzer or FailureAnalyzer()
        self.mutation_planner = mutation_planner or MutationPlanner()

    def diagnose(self, scenario: Scenario) -> TaskDiagnosis:
        return self.scenario_interpreter.interpret(scenario)

    def plan(self, **kwargs) -> MutationPlan:
        return self.mutation_planner.propose_mutations(**kwargs)

    def attach_diagnosis(self, genome: Genome, diagnosis: TaskDiagnosis, scenario: Scenario) -> Genome:
        updated = genome.model_copy(deep=True)
        updated.name = f"{scenario.name}_seed"
        updated.scenario_name = scenario.name
        updated.task_diagnosis = diagnosis.model_dump(mode="json")
        updated.stop_rule.update(
            {
                "max_generations": scenario.evolution.max_generations,
                "patience": scenario.evolution.patience,
                "min_delta": scenario.evolution.min_delta,
            }
        )
        return Genome.model_validate(updated.model_dump(mode="json"))
