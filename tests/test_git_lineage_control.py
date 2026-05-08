import json
import subprocess
from pathlib import Path

from stemos.config import Settings
from stemos.evolution.control import EvolutionControl
from stemos.evolution.loop import EvolutionLoop
from stemos.genome.loader import load_genome
from stemos.genome.serializer import save_genome
from stemos.kernel.versioning import GitProvenance, GitRunConfig


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _init_git_repo(path: Path) -> None:
    _run(["git", "init"], path)
    _run(["git", "config", "user.email", "stemos@example.test"], path)
    _run(["git", "config", "user.name", "stem_agent Test"], path)
    (path / "README.md").write_text("test repo\n", encoding="utf-8")
    _run(["git", "add", "README.md"], path)
    result = _run(["git", "commit", "-m", "initial"], path)
    assert result.returncode == 0, result.stderr


def test_git_branch_created_when_enabled(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "git_branch"
    run_dir.mkdir(parents=True)

    result = GitProvenance(
        GitRunConfig(run_id="git_branch", run_dir=run_dir, branch_enabled=True)
    ).start()
    branch = _run(["git", "branch", "--show-current"], tmp_path).stdout.strip()

    assert result.allowed is True
    assert branch == "stem_agent/run/git_branch"


def test_git_commit_created_on_promoted_mutation(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "git_commit"
    run_dir.mkdir(parents=True)
    (run_dir / "lineage.jsonl").write_text(
        json.dumps(
            {
                "event": "mutation_promoted",
                "mutation_type": "add_workflow_step",
                "target": "workflow",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    git = GitProvenance(
        GitRunConfig(
            run_id="git_commit",
            run_dir=run_dir,
            branch_enabled=True,
            commit_enabled=True,
        )
    )
    git.start()
    result = git.commit_safe("stem_agent promoted add_workflow_step", [run_dir])
    log = _run(["git", "log", "--oneline", "--max-count=2"], tmp_path).stdout

    assert result.allowed is True
    assert "stem_agent promoted add_workflow_step" in log


def test_git_push_disabled_by_default(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "git_push"
    run_dir.mkdir(parents=True)

    git = GitProvenance(GitRunConfig(run_id="git_push", run_dir=run_dir))

    result = git.push()

    assert result.allowed is False
    assert "disabled" in result.reason


def test_pause_command_writes_control_file(tmp_path):
    control = EvolutionControl(tmp_path / "runs" / "pause_001", "pause_001")

    state = control.write_command("pause")
    saved = json.loads(control.control_path.read_text(encoding="utf-8"))

    assert state.command == "pause"
    assert saved["command"] == "pause"
    assert saved["status"] == "REQUESTED"


def test_freeze_now_uses_best_verified_genome(tmp_path):
    loop = EvolutionLoop(settings=Settings(offline_mode=True), runs_root=tmp_path / "runs")
    result = loop.evolve("scenarios/toy_structured_answer", "freeze_best")
    control = EvolutionControl(result.run_dir, "freeze_best")
    state = control.load_checkpoint()
    current = load_genome(state.current_genome_path)
    current.name = "unverified_candidate_should_not_freeze"
    save_genome(current, state.current_genome_path)

    frozen = loop.freeze_now("freeze_best")
    frozen_genome = load_genome(frozen.frozen_genome_path)

    assert frozen.status == "FROZEN"
    assert frozen_genome.name != "unverified_candidate_should_not_freeze"


def test_abort_does_not_promote_candidate(tmp_path):
    run_id = "abort_001"
    run_dir = tmp_path / "runs" / run_id
    EvolutionControl(run_dir, run_id).write_command("abort")
    loop = EvolutionLoop(settings=Settings(offline_mode=True), runs_root=tmp_path / "runs")

    result = loop.evolve("scenarios/toy_structured_answer", run_id)
    lineage = (run_dir / "lineage.jsonl").read_text(encoding="utf-8")

    assert result.status == "ABORTED"
    assert "mutation_promoted" not in lineage
    assert not (run_dir / "frozen_genome.yaml").exists()


def test_secret_scan_blocks_commit_with_api_key(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text(
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456\n",
        encoding="utf-8",
    )
    _run(["git", "add", "secret.txt"], tmp_path)

    git = GitProvenance(
        GitRunConfig(run_id="secret", run_dir=tmp_path / "runs" / "secret")
    )
    git.start()
    result = git.scan_staged_files()

    assert result.allowed is False
    assert "secret" in result.reason.lower() or "OPENAI_API_KEY" in result.reason
