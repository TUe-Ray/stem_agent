from pathlib import Path


def test_gitignore_keeps_only_curated_demo_run_visible():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "runs/*" in gitignore
    assert "!runs/.gitkeep" in gitignore
    assert "!runs/demo_001/" in gitignore
    assert "!runs/demo_001/**" in gitignore
    assert "runs/demo_001/visuals/training_progress.md" in gitignore
    assert "openai_002_review_bundle.tar.gz" in gitignore
