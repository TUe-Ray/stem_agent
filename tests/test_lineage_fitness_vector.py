from stem_agent.evolution.lineage import LineageLog


def test_mutation_plan_records_fitness_vector(tmp_path):
    lineage = LineageLog(tmp_path)
    lineage.record_mutation_plan(
        generation=1,
        summary="test",
        mutation_count=2,
        fitness_vector={"task_quality": 0.7, "safety_score": 1.0},
    )
    content = (tmp_path / "lineage.jsonl").read_text(encoding="utf-8")
    assert '"fitness_vector"' in content
    assert '"task_quality": 0.7' in content
