from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stem_agent.genome.models import Genome
from stem_agent.genome.serializer import save_genome
from stem_agent.kernel.sandbox import Sandbox
from stem_agent.kernel.validators import ValidationResult


class VersionStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.snapshots_dir = run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_genome(self, version_id: str, genome: Genome) -> Path:
        path = self.snapshots_dir / f"{version_id}.yaml"
        save_genome(genome, path)
        return path

    def rollback_artifacts(self, version_id: str, target_path: Path) -> None:
        source = self.snapshots_dir / f"{version_id}.yaml"
        if not source.exists():
            raise FileNotFoundError(f"Missing snapshot: {source}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_path)


class GenomeArchive:
    def __init__(
        self,
        max_size: int = 10,
        *,
        run_dir: str | Path | None = None,
        entries: list[dict[str, Any]] | None = None,
        load_existing: bool = True,
        reset: bool = False,
    ):
        self.max_size = max(1, int(max_size))
        self.run_dir = Path(run_dir) if run_dir else None
        self.entries: list[dict[str, Any]] = []
        if reset and self.archive_path:
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            self.archive_path.write_text("", encoding="utf-8")
        if entries:
            for entry in entries:
                self._add_entry(dict(entry), persist=False)
        elif load_existing and self.archive_path and self.archive_path.exists():
            self._load_from_jsonl(self.archive_path)

    @property
    def archive_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "archive.jsonl"

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any] | None,
        *,
        run_dir: str | Path,
        max_size: int = 10,
    ) -> "GenomeArchive":
        if snapshot and snapshot.get("entries"):
            return cls(max_size=int(snapshot.get("max_size", max_size)), run_dir=run_dir, entries=snapshot["entries"])
        return cls(max_size=max_size, run_dir=run_dir)

    def add(
        self,
        genome: dict,
        score: float,
        generation: int,
        parent_id: str | None,
        *,
        fitness_vector: dict[str, float] | None = None,
    ) -> str:
        genome_id = str(uuid.uuid4())
        entry = {
            "genome_id": genome_id,
            "genome": genome,
            "score": float(score),
            "generation": int(generation),
            "parent_id": parent_id,
            "fitness_vector": fitness_vector or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._add_entry(entry, persist=True)
        return genome_id

    def sample_parent(self, strategy: str = "weighted") -> tuple[str, dict]:
        if not self.entries:
            raise RuntimeError("Cannot sample parent from an empty genome archive")
        if strategy == "best":
            entry = max(self.entries, key=lambda item: float(item["score"]))
        elif strategy == "random":
            entry = random.choice(self.entries)
        elif strategy == "weighted":
            weights = self._softmax_weights([float(item["score"]) for item in self.entries])
            entry = random.choices(self.entries, weights=weights, k=1)[0]
        else:
            raise ValueError(f"Unsupported archive sampling strategy: {strategy}")
        return str(entry["genome_id"]), dict(entry["genome"])

    def best(self) -> tuple[str, dict, float]:
        if not self.entries:
            raise RuntimeError("Cannot select best genome from an empty archive")
        entry = max(self.entries, key=lambda item: float(item["score"]))
        return str(entry["genome_id"]), dict(entry["genome"]), float(entry["score"])

    def score_for(self, genome_id: str) -> float:
        for entry in self.entries:
            if entry.get("genome_id") == genome_id:
                return float(entry.get("score", 0.0))
        raise KeyError(genome_id)

    def to_markdown_table(self) -> str:
        rows = [
            "| Rank | Genome ID | Score | Task | CostEff | Safety | Stability | Generation | Parent ID |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for rank, entry in enumerate(
            sorted(self.entries, key=lambda item: float(item["score"]), reverse=True),
            start=1,
        ):
            parent_id = entry.get("parent_id") or ""
            fitness = entry.get("fitness_vector", {}) or {}
            rows.append(
                "| {rank} | `{genome_id}` | {score:.4f} | {task} | {cost_eff} | {safety} | {stability} | {generation} | {parent} |".format(
                    rank=rank,
                    genome_id=entry.get("genome_id", ""),
                    score=float(entry.get("score", 0.0)),
                    task=f"{float(fitness.get('task_quality', 0.0)):.4f}" if fitness else "-",
                    cost_eff=f"{float(fitness.get('cost_efficiency', 0.0)):.4f}" if fitness else "-",
                    safety=f"{float(fitness.get('safety_score', 0.0)):.4f}" if fitness else "-",
                    stability=f"{float(fitness.get('stability_score', 0.0)):.4f}" if fitness else "-",
                    generation=entry.get("generation", ""),
                    parent=f"`{parent_id}`" if parent_id else "",
                )
            )
        if len(rows) == 2:
            rows.append("| | | | | | | | | Archive is empty. |")
        return "\n".join(rows)

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_size": self.max_size,
            "entries": [dict(entry) for entry in self.entries],
        }

    def __len__(self) -> int:
        return len(self.entries)

    def _add_entry(self, entry: dict[str, Any], *, persist: bool) -> None:
        self.entries.append(entry)
        self.entries.sort(key=lambda item: float(item["score"]), reverse=True)
        if len(self.entries) > self.max_size:
            self.entries = self.entries[: self.max_size]
        if persist and self.archive_path:
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            with self.archive_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _load_from_jsonl(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "genome_id" in item and "genome" in item and "score" in item:
                if "fitness_vector" not in item:
                    item["fitness_vector"] = {}
                self._add_entry(item, persist=False)

    def _softmax_weights(self, scores: list[float]) -> list[float]:
        peak = max(scores)
        raw = [math.exp(score - peak) for score in scores]
        total = sum(raw)
        if total <= 0:
            return [1.0 for _ in scores]
        return [value / total for value in raw]


SECRET_PATTERNS = [
    re.compile(r"OPENAI_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}", re.IGNORECASE),
]


@dataclass
class GitRunConfig:
    run_id: str
    run_dir: Path
    branch_enabled: bool = False
    commit_enabled: bool = False
    push_enabled: bool = False
    remote: str = "origin"
    remote_allowlist: tuple[str, ...] = ("origin",)
    max_file_bytes: int = 1_000_000

    @property
    def branch_name(self) -> str:
        return f"stem_agent/run/{self.run_id}"


@dataclass
class GitProvenance:
    """Git provenance lives in the immutable kernel/versioning boundary."""

    config: GitRunConfig
    repo_root: Path | None = None
    _started: bool = field(default=False, init=False)

    def start(self) -> ValidationResult:
        if not (self.config.branch_enabled or self.config.commit_enabled or self.config.push_enabled):
            return ValidationResult.allow("git provenance disabled")
        repo = self._repo_root()
        if repo is None:
            return ValidationResult.reject("Git provenance requested outside a git repository")
        self.repo_root = repo
        if self.config.branch_enabled:
            result = self._git(["checkout", "-B", self.config.branch_name])
            if result.returncode != 0:
                return ValidationResult.reject(result.stderr.strip() or "failed to create git branch")
        self._started = True
        return ValidationResult.allow("git provenance ready")

    def commit_safe(self, message: str, paths: list[str | Path] | None = None) -> ValidationResult:
        if not self.config.commit_enabled:
            return ValidationResult.allow("git commit disabled")
        start_result = self.start() if not self._started else ValidationResult.allow("git started")
        if not start_result.allowed:
            return start_result

        repo = self.repo_root or self._repo_root()
        if repo is None:
            return ValidationResult.reject("Cannot commit outside a git repository")
        paths = paths or [self.config.run_dir]
        relative_paths = [self._relative_path(repo, Path(path)) for path in paths]

        add_result = self._git(["add", "--", *relative_paths])
        if add_result.returncode != 0:
            return ValidationResult.reject(add_result.stderr.strip() or "git add failed")

        scan_result = self.scan_staged_files()
        if not scan_result.allowed:
            return scan_result

        diff_result = self._git(["diff", "--cached", "--quiet", "--", *relative_paths])
        if diff_result.returncode == 0:
            return ValidationResult.allow("nothing to commit")

        commit_result = self._git(["commit", "-m", message, "--", *relative_paths])
        if commit_result.returncode != 0:
            return ValidationResult.reject(commit_result.stderr.strip() or "git commit failed")
        return ValidationResult.allow("git checkpoint committed")

    def push(self) -> ValidationResult:
        if not self.config.push_enabled:
            return ValidationResult.reject("git push disabled; pass --git-push or use push-run")
        start_result = self.start() if not self._started else ValidationResult.allow("git started")
        if not start_result.allowed:
            return start_result
        allow_result = self.validate_remote_allowed(self.config.remote)
        if not allow_result.allowed:
            return allow_result
        scan_result = self.scan_staged_files()
        if not scan_result.allowed:
            return scan_result
        push_result = self._git(
            ["push", "-u", self.config.remote, self.config.branch_name]
        )
        if push_result.returncode != 0:
            return ValidationResult.reject(push_result.stderr.strip() or "git push failed")
        return ValidationResult.allow("git run branch pushed")

    def scan_staged_files(self) -> ValidationResult:
        if self.repo_root is None:
            self.repo_root = self._repo_root()
        if self.repo_root is None:
            return ValidationResult.reject("Cannot scan outside a git repository")
        listed = self._git(["diff", "--cached", "--name-only", "-z"])
        if listed.returncode != 0:
            return ValidationResult.reject(listed.stderr.strip() or "git staged-file scan failed")
        names = [name for name in listed.stdout.split("\0") if name]
        for name in names:
            path = self.repo_root / name
            if Path(name).name == ".env" or name.endswith("/.env"):
                return ValidationResult.reject("Refusing to commit .env")
            if not path.exists() or path.is_dir():
                continue
            if path.stat().st_size > self.config.max_file_bytes:
                return ValidationResult.reject(f"Refusing to commit large temporary file: {name}")
            text = self._read_text_for_scan(path)
            if "OPENAI_API_KEY" in text and "=" in text and not text.strip().endswith("OPENAI_API_KEY="):
                return ValidationResult.reject(f"Potential OPENAI_API_KEY secret in staged file: {name}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    return ValidationResult.reject(f"Potential secret pattern in staged file: {name}")
            generated_validation = self._validate_generated_tool_artifact(name, text)
            if not generated_validation.allowed:
                return generated_validation
        return ValidationResult.allow("staged files passed secret scan")

    def validate_remote_allowed(self, remote: str) -> ValidationResult:
        if remote not in self.config.remote_allowlist:
            return ValidationResult.reject(f"Remote is not allowlisted: {remote}")
        remote_result = self._git(["remote", "get-url", remote])
        if remote_result.returncode != 0:
            return ValidationResult.reject(remote_result.stderr.strip() or f"Unknown remote: {remote}")
        return ValidationResult.allow("remote passed allowlist check")

    def _repo_root(self) -> Path | None:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return Path(result.stdout.strip())

    def _git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root or Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )

    def _relative_path(self, repo: Path, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.exists():
            resolved = path.absolute()
        try:
            return str(resolved.relative_to(repo.resolve()))
        except ValueError:
            return str(path)

    def _read_text_for_scan(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""

    def _validate_generated_tool_artifact(self, name: str, text: str) -> ValidationResult:
        if "generated_tools/" not in name or not name.endswith(".py"):
            return ValidationResult.allow("not a generated tool artifact")
        validation = Sandbox().validate_generated_tool_code(text)
        if validation.allowed:
            return validation
        if name.startswith("runs/"):
            parts = Path(name).parts
            if len(parts) >= 2:
                lineage_path = self.repo_root / parts[0] / parts[1] / "lineage.jsonl"
                if lineage_path.exists() and "mutation_rejected" in lineage_path.read_text(encoding="utf-8"):
                    return ValidationResult.allow("unsafe generated tool is stored with rejected lineage")
        return ValidationResult.reject(
            f"Unsafe generated tool cannot be committed without rejected lineage: {name}"
        )
