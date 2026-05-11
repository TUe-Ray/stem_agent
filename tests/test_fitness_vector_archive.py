from stem_agent.kernel.versioning import GenomeArchive


def test_archive_persists_fitness_vector(tmp_path):
    archive = GenomeArchive(run_dir=tmp_path)
    archive.add(
        genome={"name": "g"},
        score=0.9,
        generation=0,
        parent_id=None,
        fitness_vector={"task_quality": 0.8, "cost_efficiency": 0.7, "safety_score": 1.0, "stability_score": 0.6},
    )
    content = (tmp_path / "archive.jsonl").read_text(encoding="utf-8")
    assert '"fitness_vector"' in content
    table = archive.to_markdown_table()
    assert "| Task | CostEff | Safety | Stability |" in table


def test_archive_reset_ignores_previous_run_entries(tmp_path):
    archive = GenomeArchive(run_dir=tmp_path)
    archive.add(genome={"name": "old"}, score=0.99, generation=0, parent_id=None)

    fresh = GenomeArchive(run_dir=tmp_path, load_existing=False, reset=True)
    fresh.add(genome={"name": "fresh"}, score=0.5, generation=0, parent_id=None)

    content = (tmp_path / "archive.jsonl").read_text(encoding="utf-8")
    assert "old" not in content
    assert "fresh" in content
    assert len(fresh) == 1
