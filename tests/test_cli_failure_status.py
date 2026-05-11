import json
from pathlib import Path

from typer.testing import CliRunner

import stem_agent.cli as cli


SCENARIO_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "toy_structured_answer"


def test_evolve_marks_run_failed_when_unhandled_exception_occurs(tmp_path, monkeypatch):
    class FailingLoop:
        def evolve(self, *args, **kwargs):
            _ = args, kwargs
            raise RuntimeError("forced evolve failure\nwith extra details")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "EvolutionLoop", FailingLoop)

    result = CliRunner().invoke(
        cli.app,
        [
            "evolve",
            str(SCENARIO_PATH),
            "--run-id",
            "cli_failed_evolve",
            "--no-progress-bar",
        ],
    )
    state = json.loads(
        (tmp_path / "runs" / "cli_failed_evolve" / "control.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.exit_code != 0
    assert state["status"] == "FAILED"
    assert state["command"] == "evolve"
    assert state["message"] == "RuntimeError: forced evolve failure"


def test_resume_marks_run_failed_when_unhandled_exception_occurs(tmp_path, monkeypatch):
    class FailingLoop:
        def resume(self, *args, **kwargs):
            _ = args, kwargs
            raise RuntimeError("forced resume failure")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "EvolutionLoop", FailingLoop)

    result = CliRunner().invoke(cli.app, ["resume", "cli_failed_resume"])
    state = json.loads(
        (tmp_path / "runs" / "cli_failed_resume" / "control.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.exit_code != 0
    assert state["status"] == "FAILED"
    assert state["command"] == "resume"
    assert state["message"] == "RuntimeError: forced resume failure"
