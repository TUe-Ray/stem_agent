from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FORBIDDEN_FIELD_NAMES = {
    "expected_output",
    "reference_notes",
    "ground_truth",
    "answer",
    "case_input",
    "case_results",
    "per_case",
    "final_holdout_cases",
    "hidden_cases",
    "external_benchmark_cases",
}


@dataclass
class SignalLeakFinding:
    path: str
    reason: str
    snippet: str = ""


class SignalLeakAuditor:
    def __init__(self, *, forbidden_case_ids: Iterable[str] = (), forbidden_case_texts: Iterable[str] = ()):
        self.forbidden_case_ids = {str(x) for x in forbidden_case_ids if x}
        self.forbidden_case_texts = {self._normalize(str(x)) for x in forbidden_case_texts if str(x).strip()}

    def assert_no_leak_in_text(self, text: str, *, path: str = "text") -> None:
        findings = self.find_text_leaks(text, path=path)
        if findings:
            raise ValueError("Signal leak detected:\n" + "\n".join(f"{f.path}: {f.reason}: {f.snippet}" for f in findings[:10]))

    def find_text_leaks(self, text: str, *, path: str = "text") -> list[SignalLeakFinding]:
        normalized = self._normalize(text)
        findings: list[SignalLeakFinding] = []
        for case_id in self.forbidden_case_ids:
            if self._normalize(case_id) in normalized:
                findings.append(SignalLeakFinding(path, f"forbidden case id `{case_id}`", case_id))
        for case_text in self.forbidden_case_texts:
            if len(case_text) >= 20 and case_text in normalized:
                findings.append(SignalLeakFinding(path, "forbidden raw case text appeared", case_text[:120]))
        return findings

    def find_json_leaks(self, data: object, *, path: str = "$") -> list[SignalLeakFinding]:
        findings: list[SignalLeakFinding] = []
        if isinstance(data, dict):
            for key, value in data.items():
                k = str(key)
                p = f"{path}.{k}"
                if k in FORBIDDEN_FIELD_NAMES:
                    findings.append(SignalLeakFinding(p, f"forbidden field `{k}`"))
                findings.extend(self.find_json_leaks(value, path=p))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                findings.extend(self.find_json_leaks(item, path=f"{path}[{i}]"))
        elif isinstance(data, str):
            findings.extend(self.find_text_leaks(data, path=path))
        return findings

    def assert_no_leak_in_json(self, data: object, *, path: str = "$") -> None:
        findings = self.find_json_leaks(data, path=path)
        if findings:
            raise ValueError("Signal leak detected:\n" + "\n".join(f"{f.path}: {f.reason}: {f.snippet}" for f in findings[:10]))

    def assert_no_leak_in_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for n, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    self.assert_no_leak_in_json(json.loads(line), path=f"{path}:{n}")
            return
        if path.suffix == ".json":
            self.assert_no_leak_in_json(json.loads(text), path=str(path))
            return
        self.assert_no_leak_in_text(text, path=str(path))

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())


def nucleus_visible_artifact_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(run_dir.glob("lineage.jsonl"))
    paths.extend(run_dir.glob("archive.jsonl"))
    paths.extend(run_dir.glob("generation_*/mutation_plan.json"))
    paths.extend(run_dir.glob("generation_*/**/mutation_plan.json"))
    for candidate in [run_dir / "llm_calls.jsonl", run_dir / "model_calls.jsonl", run_dir / "nucleus_calls.jsonl"]:
        if candidate.exists():
            paths.append(candidate)
    return [path for path in paths if path.exists()]
