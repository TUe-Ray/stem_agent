from __future__ import annotations

from stemos.config import Settings
from stemos.evolution.loop import EvolutionLoop
from stemos.evolution.visuals import VisualizationBuilder


def test_visualize_generates_reviewer_visuals(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )
    result = loop.evolve("scenarios/toy_structured_answer", "visual_001")

    paths = VisualizationBuilder().visualize(result.run_dir)
    names = {path.name for path in paths}
    report = result.report_path.read_text(encoding="utf-8")

    assert "evolution_timeline.md" in names
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


def test_aggregate_compares_runs(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
        runs_root=tmp_path / "runs",
    )
    first = loop.evolve("scenarios/toy_structured_answer", "aggregate_001")
    second = loop.evolve("scenarios/tiny_task_operator", "aggregate_002")

    path = VisualizationBuilder().aggregate([first.run_dir, second.run_dir])
    content = path.read_text(encoding="utf-8")

    assert "StemOS Aggregate Run Report" in content
    assert "aggregate_001" in content
    assert "aggregate_002" in content
    assert "Final organism shape" in content


def test_output_comparison_uses_specific_manager_deadline_sample(tmp_path):
    loop = EvolutionLoop(
        settings=Settings(offline_mode=True),
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
