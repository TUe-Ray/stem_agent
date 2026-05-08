from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from stemos.nucleus.schemas import MutationProposal


class AWMBridge:
    def analyze(self, run_id: str, config: dict[str, Any] | None = None) -> list[MutationProposal]:
        config = config or {}
        min_occurrences = int(config.get("min_occurrences", 3))
        run_dir = Path(config.get("run_dir", Path("runs") / run_id))
        log_path = run_dir / "intra_adaptation_log.jsonl"
        if not log_path.exists():
            return []
        existing_candidate_ids = self._existing_candidate_ids(run_dir / "awm_candidates.jsonl")

        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            label = self._normalize_label(str(record.get("reflection_text", "")))
            key = (
                str(record.get("step_name", "")),
                str(record.get("quality_gate_name", "")),
                label,
            )
            groups[key].append(record)

        candidates: list[dict[str, Any]] = []
        proposals: list[MutationProposal] = []
        for (step_name, gate_name, label), records in sorted(groups.items()):
            task_ids = sorted({str(record.get("task_id", "")) for record in records if record.get("task_id")})
            if len(task_ids) < min_occurrences:
                continue
            candidate_id = self._candidate_id(run_id, step_name, gate_name, label)
            if candidate_id in existing_candidate_ids:
                continue
            pattern_summary = (
                f"{step_name} repeatedly failed {gate_name}; corrective label: {label}."
            )
            proposed = self._proposed_mutation(step_name, gate_name, label, pattern_summary)
            candidate = {
                "candidate_id": candidate_id,
                "origin": "awm_bridge",
                "mutation_type": "awm_promoted",
                "step_name": step_name,
                "quality_gate_name": gate_name,
                "pattern_summary": pattern_summary,
                "proposed_genome_fragment": proposed.patch,
                "justification": "Recurring intra-test reflection improved or attempted to improve failed steps.",
                "supporting_task_ids": task_ids,
                "occurrence_count": len(task_ids),
            }
            candidates.append(candidate)
            proposals.append(
                MutationProposal(
                    mutation_type="awm_promoted",
                    target=proposed.target,
                    rationale=pattern_summary,
                    expected_improvement=proposed.expected_improvement,
                    risk=proposed.risk,
                    origin="awm_bridge",
                    patch={
                        "proposed_mutation": proposed.model_dump(mode="json", exclude_none=True),
                        "candidate_id": candidate_id,
                        "pattern_summary": pattern_summary,
                        "supporting_task_ids": task_ids,
                    },
                )
            )

        if candidates:
            with (run_dir / "awm_candidates.jsonl").open("a", encoding="utf-8") as handle:
                for candidate in candidates:
                    handle.write(json.dumps(candidate, sort_keys=True) + "\n")
        return proposals

    def _existing_candidate_ids(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        ids: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("candidate_id"):
                ids.add(str(item["candidate_id"]))
        return ids

    def _proposed_mutation(
        self, step_name: str, gate_name: str, label: str, pattern_summary: str
    ) -> MutationProposal:
        if "retry" in label or "try" in label:
            return MutationProposal(
                mutation_type="modify_retry_policy",
                target="retry_policy",
                rationale=pattern_summary,
                expected_improvement="Give the harness one conservative retry when this failure pattern appears.",
                risk="Retries add cost and may repeat weak outputs.",
                patch={"max_attempts": 2, "revise_on_failure": True},
            )
        if "gate" in label or "check" in label:
            return MutationProposal(
                mutation_type="add_quality_gate",
                target="quality_gates",
                rationale=pattern_summary,
                expected_improvement="Make the recurring quality issue explicit before final delivery.",
                risk="The gate may be too broad if the reflection pattern is noisy.",
                patch={
                    "name": self._safe_name(f"awm_{gate_name or step_name}_gate"),
                    "description": f"Check recurring issue from AWM pattern: {label}.",
                    "check_type": "manual_placeholder",
                    "required": True,
                },
            )
        return MutationProposal(
            mutation_type="add_workflow_step",
            target="workflow",
            rationale=pattern_summary,
            expected_improvement="Turn recurring corrective reflection into an explicit review step.",
            risk="Adds one workflow step and may increase complexity.",
            patch={
                "id": self._safe_name(f"awm_review_{step_name or 'step'}"),
                "role": "Founder",
                "action": f"Apply corrective instruction before final output: {label}.",
                "input_from": [step_name] if step_name else [],
                "output_key": "awm_review_notes",
            },
        )

    def _normalize_label(self, text: str) -> str:
        lowered = " ".join(text.lower().split())
        if any(word in lowered for word in ["retry", "try again", "revise"]):
            return "retry_with_revision"
        if any(word in lowered for word in ["missing", "section", "requirement"]):
            return "cover_missing_requirements"
        if any(word in lowered for word in ["format", "schema", "structure"]):
            return "fix_output_structure"
        return lowered[:80] or "unspecified_correction"

    def _candidate_id(self, run_id: str, step_name: str, gate_name: str, label: str) -> str:
        digest = hashlib.sha256(f"{run_id}:{step_name}:{gate_name}:{label}".encode("utf-8")).hexdigest()[:16]
        return f"awm_{digest}"

    def _safe_name(self, raw: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()
        if not safe:
            return "awm_candidate"
        if safe[0].isdigit():
            safe = f"awm_{safe}"
        return safe[:64]
