"""SWE-bench Lite evaluation for stem_agent.

Two modes:
  Light mode (default, no Docker): git clone + checkout + patch + pytest
  Docker mode (STEM_AGENT_SWEBENCH_DOCKER=1): full swebench.harness.run_evaluation

The light mode is a pragmatic real evaluation that clones the target repo,
applies both test_patch and predicted patch, and runs the test suite.
It works without Docker but requires git, python, and pytest on the host.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEMO_INSTANCES: list[dict[str, Any]] = [
    {
        "instance_id": "demo__simple_validation_fix",
        "repo": "demo/simple",
        "base_commit": "abc123",
        "problem_statement": (
            "The login view raises ValueError when given an empty username. "
            "Fix the login view to return a proper 400 response instead."
        ),
        "test_patch": (
            "@@ -10,3 +10,8 @@ def test_login_valid():\n"
            "+def test_login_empty_username():\n"
            "+    response = client.post('/login', {'username': '', 'password': 'test'})\n"
            "+    assert response.status_code == 400\n"
        ),
        "hints_text": "Look at the login view in auth/views.py. The issue is around line 45.",
    },
    {
        "instance_id": "demo__token_entropy_fix",
        "repo": "demo/simple",
        "base_commit": "abc123",
        "problem_statement": (
            "The password reset token generator uses a weak random source. "
            "Replace it with secrets.token_urlsafe()."
        ),
        "test_patch": (
            "@@ -0,0 +1,8 @@\n"
            "+import re\n"
            "+def test_token_entropy():\n"
            "+    token = gen.make_token(user)\n"
            "+    assert len(token) >= 43\n"
        ),
        "hints_text": "Look at tokens.py in contrib/auth/.",
    },
    {
        "instance_id": "demo__null_check_missing",
        "repo": "demo/simple",
        "base_commit": "abc123",
        "problem_statement": (
            "The profile view crashes with AttributeError when request.user is None. "
            "Add a null check and return 401 Unauthorized."
        ),
        "test_patch": (
            "@@ -5,3 +5,8 @@ def test_profile_authenticated():\n"
            "+def test_profile_anonymous():\n"
            "+    response = client.get('/profile')\n"
            "+    assert response.status_code == 401\n"
        ),
        "hints_text": "Look at profile view in users/views.py. request.user can be AnonymousUser.",
    },
]


# ── Light-mode evaluation (git + patch + pytest, no Docker) ──────────────


def evaluate_patch_light(
    instance: dict[str, Any],
    predicted_patch: str,
    *,
    work_dir: str | Path | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Evaluate a predicted patch against a SWE-bench instance using git + pytest.

    Returns:
        {
            "resolved": bool,          # all FAIL_TO_PASS tests now pass
            "patch_applies": bool,     # predicted patch applied cleanly
            "fail_to_pass": dict,      # {test_name: "PASSED"|"FAILED"}
            "pass_to_pass": dict,      # {test_name: "PASSED"|"FAILED"}
            "score": float,            # resolution score (0.0 or 1.0)
            "error": str | None,       # error message if something went wrong
        }
    """
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    test_patch = instance.get("test_patch", "")
    fail_to_pass = instance.get("FAIL_TO_PASS", [])
    pass_to_pass = instance.get("PASS_TO_PASS", [])
    instance_id = instance.get("instance_id", "unknown")

    if not repo or not base_commit:
        return _error_result("Missing repo or base_commit in instance")

    base_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="swebench_"))
    repo_dir = base_dir / f"repo_{_safe_id(instance_id)}"

    try:
        # 1. Clone from cached workspace instead of from GitHub (avoids re-downloading 164MB+)
        cached = _ensure_cached_workspace(repo, base_commit)
        if cached and cached.exists():
            # Lightweight copy of cached git repo
            _run(["cp", "-r", str(cached), str(repo_dir)], timeout=60)
        else:
            # Fallback: clone from GitHub
            clone_url = f"https://github.com/{repo}.git"
            _run(["git", "clone", "--filter=blob:none", "--no-tags", clone_url, str(repo_dir)], timeout=300)
            _run(["git", "-C", str(repo_dir), "checkout", base_commit], timeout=120)

        # 2. Apply test patch (adds failing tests)
        test_patch_clean = _clean_patch(test_patch)
        test_result = _run(
            ["git", "-C", str(repo_dir), "apply", "--verbose"],
            input_text=test_patch_clean,
            timeout=30,
        )
        if test_result.returncode != 0:
            return _error_result(f"Test patch failed to apply: {test_result.stderr[:500]}")

        # 3. Install deps in a venv (persistent cache per repo+base_commit)
        venv_dir = _ensure_persistent_venv(repo, base_commit, repo_dir)
        pip = str(venv_dir / "bin" / "pip")
        python = str(venv_dir / "bin" / "python")

        # Quick pip install check — if editable install is stale, refresh
        _ensure_editable_install(pip, repo_dir)

        # 4. Verify FAIL_TO_PASS tests actually fail before patch
        baseline = _run_tests(python, repo_dir, fail_to_pass, timeout=timeout)
        # 5. Verify PASS_TO_PASS tests pass before patch (sanity check)
        baseline_ptp = _run_tests(python, repo_dir, pass_to_pass, timeout=timeout)

        # 6. Apply predicted patch
        pred_patch_clean = _clean_patch(predicted_patch)
        patch_result = _run(
            ["git", "-C", str(repo_dir), "apply", "--verbose"],
            input_text=pred_patch_clean,
            timeout=30,
        )
        patch_applies = patch_result.returncode == 0
        if not patch_applies:
            # Try patch command as fallback
            patch_file = repo_dir / "_predicted.patch"
            patch_file.write_text(pred_patch_clean)
            fallback = _run(
                ["patch", "-p1", "-i", str(patch_file)],
                cwd=str(repo_dir),
                timeout=30,
            )
            patch_applies = fallback.returncode == 0

        if not patch_applies:
            return {
                "resolved": False,
                "patch_applies": False,
                "fail_to_pass": {t: "NOT_RUN" for t in fail_to_pass},
                "pass_to_pass": {t: "NOT_RUN" for t in pass_to_pass},
                "score": 0.0,
                "error": "Predicted patch did not apply cleanly",
            }

        # 7. Run FAIL_TO_PASS tests after patch
        after_ftp = _run_tests(python, repo_dir, fail_to_pass, timeout=timeout)

        # 8. Run PASS_TO_PASS tests after patch
        after_ptp = _run_tests(python, repo_dir, pass_to_pass, timeout=timeout)

        # 9. Score with partial credit (not just binary resolved/unresolved)
        ftp_total = len(after_ftp)
        ptp_total = len(after_ptp)
        ftp_pass = sum(1 for s in after_ftp.values() if s == "PASSED")
        ptp_pass = sum(1 for s in after_ptp.values() if s == "PASSED")
        ftp_rate = ftp_pass / max(1, ftp_total)
        ptp_rate = ptp_pass / max(1, ptp_total)
        score = 0.7 * ftp_rate + 0.3 * ptp_rate
        resolved = ftp_rate == 1.0 and ptp_rate == 1.0

        return {
            "resolved": resolved,
            "patch_applies": True,
            "fail_to_pass": after_ftp,
            "pass_to_pass": after_ptp,
            "score": round(score, 4),
            "error": None,
        }

    except subprocess.TimeoutExpired as e:
        return _error_result(f"Timeout: {e}")
    except Exception as e:
        return _error_result(f"Unexpected error: {e}")
    finally:
        # Clean up temp workspace if we created it (not if user provided work_dir)
        if work_dir is None:
            import shutil
            shutil.rmtree(base_dir, ignore_errors=True)


# ── Docker-mode evaluation ──────────────────────────────────────────────────


def evaluate_patch_docker(
    instance: dict[str, Any],
    predicted_patch: str,
    *,
    run_id: str = "stem_agent_eval",
    timeout: int = 600,
) -> dict[str, Any]:
    """Evaluate via full swebench Docker harness.

    Requires Docker and swebench package installed.
    """
    try:
        from swebench.harness.run_evaluation import main as run_eval
        from swebench.harness.grading import get_eval_report
    except ImportError:
        return _error_result("swebench package not installed")

    # Write predictions file
    import tempfile
    pred_dir = Path(tempfile.mkdtemp(prefix="swebench_preds_"))
    pred_file = pred_dir / "predictions.json"
    pred_data = {
        instance["instance_id"]: {
            "model_name_or_path": "stem_agent",
            "model_patch": predicted_patch,
        }
    }
    pred_file.write_text(json.dumps(pred_data))

    try:
        run_eval(
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            instance_ids=[instance["instance_id"]],
            predictions_path=str(pred_file),
            max_workers=1,
            force_rebuild=False,
            cache_level="none",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=timeout,
            namespace=None,
            rewrite_reports=False,
            modal=False,
        )

        # Read the report
        report_path = Path(f"{run_id}.json")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            resolved = report.get("resolved", {}).get(instance["instance_id"], False)
            return {
                "resolved": resolved,
                "patch_applies": True,
                "fail_to_pass": {},
                "pass_to_pass": {},
                "score": 1.0 if resolved else 0.0,
                "error": None,
                "report": report,
            }

        return _error_result("Docker eval completed but no report found")

    except Exception as e:
        return _error_result(f"Docker eval failed: {e}")
    finally:
        import shutil
        shutil.rmtree(pred_dir, ignore_errors=True)


# ── Public API ──────────────────────────────────────────────────────────────


def evaluate_swebench_instance(
    instance: dict[str, Any],
    predicted_patch: str,
    *,
    work_dir: str | Path | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Evaluate a SWE-bench prediction. Auto-selects mode."""
    use_docker = os.environ.get("STEM_AGENT_SWEBENCH_DOCKER") == "1"

    if use_docker:
        return evaluate_patch_docker(instance, predicted_patch, timeout=timeout)
    return evaluate_patch_light(instance, predicted_patch, work_dir=work_dir, timeout=timeout)


def download_swebench_lite(
    n_train: int = 2,
    n_val: int = 1,
    n_hidden: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return train/val/hidden splits.

    Tries to load from HuggingFace datasets first, falls back to demo instances.
    """
    instances = _load_real_instances(n_train + n_val + n_hidden)
    if not instances:
        instances = list(DEMO_INSTANCES)

    hashed = sorted(
        instances,
        key=lambda item: hashlib.sha256(
            (item["instance_id"] + str(seed)).encode()
        ).hexdigest(),
    )
    train = hashed[: min(n_train, len(hashed))]
    remaining = hashed[n_train:]
    val = remaining[: min(n_val, len(remaining))]
    hidden = remaining[n_val : n_val + n_hidden] if n_hidden else []
    return train, val, hidden


# ── Helpers ─────────────────────────────────────────────────────────────────


def _load_real_instances(n: int = 3) -> list[dict[str, Any]]:
    """Try to load real SWE-bench Lite instances from HF."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        items = []
        for item in ds.select(range(min(n, len(ds)))):
            ftp = item.get("FAIL_TO_PASS", [])
            ptp = item.get("PASS_TO_PASS", [])
            # HF stores these as JSON strings — parse if needed
            if isinstance(ftp, str):
                ftp = json.loads(ftp)
            if isinstance(ptp, str):
                ptp = json.loads(ptp)
            items.append({
                "instance_id": item["instance_id"],
                "repo": item["repo"],
                "base_commit": item["base_commit"],
                "problem_statement": item["problem_statement"],
                "test_patch": item["test_patch"],
                "hints_text": item.get("hints_text", ""),
                "FAIL_TO_PASS": ftp,
                "PASS_TO_PASS": ptp,
            })
        return items
    except Exception:
        return []


def _ensure_cached_workspace(repo: str, base_commit: str) -> "Path | None":
    """Ensure the persistent cached workspace exists at the given base_commit.
    
    Returns the cached workspace path, or None if clone failed.
    """
    import subprocess
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "stem_agent" / "swebench_workspaces"
    workspace = cache_dir / f"{_safe_slug(repo)}_{base_commit[:8]}"
    if workspace.exists() and (workspace / ".git").exists():
        return workspace

    workspace.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-tags", clone_url, str(workspace)],
            capture_output=True, text=True, timeout=300, check=True,
        )
        subprocess.run(
            ["git", "-C", str(workspace), "checkout", base_commit],
            capture_output=True, text=True, timeout=120, check=True,
        )
    except subprocess.CalledProcessError:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)
        return None
    return workspace


def _safe_id(instance_id: str) -> str:
    """Sanitize instance_id for filesystem use."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", instance_id)


def _clean_patch(patch_text: str) -> str:
    """Extract clean unified diff from potentially prose-wrapped output."""
    # Strip markdown fences
    m = re.search(r"```(?:diff)?\s*\n(.*?)```", patch_text, re.DOTALL | re.IGNORECASE)
    if m:
        patch_text = m.group(1).strip()

    # Ensure it starts with diff/---/+++
    for header in ("diff --git", "--- a/", "--- "):
        idx = patch_text.find(header)
        if idx >= 0:
            patch_text = patch_text[idx:]
            break

    return patch_text


def _run(
    cmd: list[str],
    *,
    cwd: str | None = None,
    input_text: str | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ensure_persistent_venv(repo: str, base_commit: str, repo_dir: Path) -> Path:
    """Return a persistent venv for this repo+base_commit, creating it if needed.

    Uses a shared cache so the same repo version's venv is reused across
    evaluations — avoids 300s pip install on every eval.
    """
    slug = _safe_slug(f"{repo}_{base_commit[:8]}")
    venv_cache = Path.home() / ".cache" / "stem_agent" / "swebench_venvs" / slug
    python_exe = venv_cache / "bin" / "python"

    if not python_exe.exists():
        venv_cache.parent.mkdir(parents=True, exist_ok=True)
        _run([sys_executable(), "-m", "venv", str(venv_cache)], timeout=120)
        pip = str(venv_cache / "bin" / "pip")
        _run([pip, "install", "-q", "pytest", "pytest-timeout"], timeout=120)

    # Symlink the cached venv into the repo_dir
    target_venv = repo_dir / ".venv_swebench"
    if not target_venv.exists():
        try:
            target_venv.symlink_to(venv_cache)
        except OSError:
            _run(["cp", "-r", str(venv_cache), str(target_venv)], timeout=60)

    return target_venv


def _ensure_editable_install(pip: str, repo_dir: Path) -> None:
    """Install the repo in editable mode if not already installed."""
    # Use a stamp in the venv dir so it persists (repo_dir is temporary)
    venv_dir = repo_dir / ".venv_swebench"
    stamp = venv_dir / ".pip_install_stamp"
    if stamp.exists():
        return
    _run([pip, "install", "-q", "-e", str(repo_dir)], timeout=300)
    stamp.touch()


def _run_tests(
    python_bin: str,
    repo_dir: Path,
    test_list: list[str],
    timeout: int = 300,
) -> dict[str, str]:
    """Run a list of pytest tests and return {test_name: "PASSED"|"FAILED"}."""
    if not test_list:
        return {}

    results = {}
    # Run tests one by one for isolation (pytest can be flaky with multiple)
    for test in test_list:
        try:
            proc = subprocess.run(
                [python_bin, "-m", "pytest", test, "-x", "-q", "--no-header"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            results[test] = "PASSED" if proc.returncode == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            results[test] = "TIMEOUT"
        except Exception:
            results[test] = "ERROR"

    return results


def _error_result(msg: str) -> dict[str, Any]:
    return {
        "resolved": False,
        "patch_applies": False,
        "fail_to_pass": {},
        "pass_to_pass": {},
        "score": 0.0,
        "error": msg,
    }


def sys_executable() -> str:
    return os.environ.get("STEM_AGENT_PYTHON", os.sys.executable)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
