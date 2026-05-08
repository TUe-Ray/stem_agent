from __future__ import annotations

from stemos.config import Settings
from stemos.evolution.loop import EvolutionLoop
from stemos.evolution.visuals import VisualizationBuilder


def test_visualize_generates_reviewer_visuals(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )
    result = loop.evolve("scenarios/toy_structured_answer", "visual_001")

    paths = VisualizationBuilder().visualize(result.run_dir)
    names = {path.name for path in paths}
    report = result.report_path.read_text(encoding="utf-8")

    assert "evolution_timeline.md" in names
    assert "training_progress.md" in names
    assert "organism_shape.md" in names
    assert "harness_before_after.md" in names
    assert "guardian_selection_board.md" in names
    assert "output_comparison.md" in names
    assert "visual_report.md" in names
    assert "```mermaid" in report
    assert "## OpenAI Run Metadata" in report
    assert "## Safe Stop And Recovery" in report
    assert "## Guardian Selection Board" in report
    assert "Roles are not predefined subagents" in report
    progress = (result.run_dir / "visuals" / "training_progress.md").read_text(
        encoding="utf-8"
    )
    assert "Promotion Score Chart" in progress
    assert "Generation Scores" in progress
    assert "Mutation Decisions" in progress


def test_aggregate_compares_runs(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )
    first = loop.evolve("scenarios/toy_structured_answer", "aggregate_001")
    second = loop.evolve("scenarios/tiny_task_operator", "aggregate_002")

    path = VisualizationBuilder().aggregate([first.run_dir, second.run_dir])
    content = path.read_text(encoding="utf-8")

    assert "stem_agent Aggregate Run Report" in content
    assert "aggregate_001" in content
    assert "aggregate_002" in content
    assert "Final organism shape" in content


def test_progress_visualization_handles_partial_run(tmp_path):
    run_dir = tmp_path / "runs" / "partial_001"
    generation_dir = run_dir / "generation_000"
    generation_dir.mkdir(parents=True)
    (run_dir / "lineage.jsonl").write_text(
        '{"event":"evaluation","generation":0,"score":0.25,"summary":"first pass"}\n',
        encoding="utf-8",
    )
    (generation_dir / "eval_result.json").write_text(
        '{"promotion_score":0.25,"train_score":0.2,"validation_score":0.3}\n',
        encoding="utf-8",
    )

    path = VisualizationBuilder().progress(run_dir)
    content = path.read_text(encoding="utf-8")

    assert path.name == "training_progress.md"
    assert "in progress or stopped before freeze" in content
    assert "| 0 | 0.2500 | 0.2000 | 0.3000 | first pass |" in content


def test_output_comparison_uses_specific_manager_deadline_sample(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(test_mode=True),
        runs_root=tmp_path / "runs",
    )
    result = loop.evolve("scenarios/toy_structured_answer", "output_compare_001")

    VisualizationBuilder().visualize(result.run_dir)
    comparison = (
        result.run_dir / "visuals" / "output_comparison.md"
    ).read_text(encoding="utf-8")

    assert "missed deadlines" in comparison
    assert "## Meeting Goal" in comparison
    assert "## Talking Points" in comparison
    assert "## Likely Objections" in comparison
    assert "## Recovery Plan" in comparison
    assert "## Wording To Use" in comparison
    assert "## Follow-Up Actions" in comparison
