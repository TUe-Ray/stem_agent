from stem_agent.genome.loader import load_default_genome
from stem_agent.nucleus.mutation_planner import MutationPlanner
from stem_agent.scenarios.loader import load_scenario


class LegacyMutationClient:
    test_mode = False

    def __init__(self):
        self.repairs = 0

    def call(self, *args, **kwargs):
        return {
            "mutation_type": "add_quality_gate",
            "target_field": "quality_gates",
            "rationale": "Low quality-gate usage needs a lightweight schema gate.",
            "new_value": {},
        }

    def audit_call(self, *args, **kwargs):
        return None

    def record_structured_output_repair(self):
        self.repairs += 1


def test_mutation_planner_canonicalizes_legacy_single_mutation_shape():
    bundle = load_scenario("scenarios/toy_structured_answer")
    genome = load_default_genome()
    client = LegacyMutationClient()
    planner = MutationPlanner(client)  # type: ignore[arg-type]

    plan = planner.propose_mutations(
        scenario=bundle.scenario,
        genome=genome,
        generation=0,
        last_score=0.5,
        best_score=0.5,
        failure_patterns=["Low quality-gate usage"],
        structured_failure_patterns=[],
        budget_remaining=1.0,
        lineage_summary="",
    )

    mutation = plan.proposed_mutations[0]
    assert mutation.mutation_type == "add_quality_gate"
    assert mutation.target == "quality_gates"
    assert mutation.patch["name"] == "required_sections_gate"
    assert mutation.patch["check_type"] == "schema"
    assert client.repairs >= 1
