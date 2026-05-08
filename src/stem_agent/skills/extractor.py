from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from stem_agent.genome.models import Genome
from stem_agent.skills.schema import AtomicSkill, SkillType


class SkillExtractor:
    def extract(
        self,
        *,
        old_genome: Genome,
        new_genome: Genome,
        mutation_type: str,
        score_before: float,
        score_after: float,
        run_id: str,
        scenario_name: str,
        generation_number: int,
        scenario_domain_tags: list[str],
    ) -> list[AtomicSkill]:
        score_lift = score_after - score_before
        old_data = old_genome.model_dump(mode="json")
        new_data = new_genome.model_dump(mode="json")
        changes = self._diff("", old_data, new_data)
        atoms: list[AtomicSkill] = []
        for field_path, fragment in changes:
            skill_type = self._infer_skill_type(field_path)
            if skill_type is None:
                continue
            name = self._skill_name(mutation_type, field_path)
            atom_id = self._skill_id(
                run_id,
                scenario_name,
                generation_number,
                mutation_type,
                field_path,
                fragment,
            )
            atoms.append(
                AtomicSkill(
                    id=atom_id,
                    name=name,
                    description=(
                        f"Reusable genome fragment affecting {field_path} "
                        f"introduced by {mutation_type}."
                    ),
                    skill_type=skill_type,
                    genome_fragment={"field_path": field_path, "value": fragment},
                    domain_tags=list(dict.fromkeys(scenario_domain_tags)),
                    origin_run_id=run_id,
                    origin_scenario_name=scenario_name,
                    origin_generation=generation_number,
                    score_lift=score_lift,
                )
            )
        return atoms

    def _diff(self, path: str, old: Any, new: Any) -> list[tuple[str, Any]]:
        if old == new:
            return []
        if old is None and isinstance(new, dict):
            return self._diff(path, {}, new)
        if isinstance(old, dict) and new is None:
            return [(path, None)]
        if isinstance(old, dict) and isinstance(new, dict):
            changes: list[tuple[str, Any]] = []
            for key in sorted(set(old) | set(new)):
                child_path = f"{path}.{key}" if path else str(key)
                changes.extend(self._diff(child_path, old.get(key), new.get(key)))
            return changes
        if isinstance(old, list) and isinstance(new, list):
            return self._diff_lists(path, old, new)
        return [(path, new)]

    def _diff_lists(self, path: str, old: list[Any], new: list[Any]) -> list[tuple[str, Any]]:
        id_field = self._list_id_field(path)
        if id_field:
            old_by_id = {
                str(item[id_field]): item
                for item in old
                if isinstance(item, dict) and item.get(id_field) is not None
            }
            new_by_id = {
                str(item[id_field]): item
                for item in new
                if isinstance(item, dict) and item.get(id_field) is not None
            }
            if len(old_by_id) == len(old) and len(new_by_id) == len(new):
                changes: list[tuple[str, Any]] = []
                for key in sorted(set(old_by_id) | set(new_by_id)):
                    item_path = f"{path}[{id_field}={key}]"
                    changes.extend(self._diff(item_path, old_by_id.get(key), new_by_id.get(key)))
                return changes
        changes = []
        max_len = max(len(old), len(new))
        for index in range(max_len):
            item_path = f"{path}[{index}]"
            old_item = old[index] if index < len(old) else None
            new_item = new[index] if index < len(new) else None
            changes.extend(self._diff(item_path, old_item, new_item))
        return changes

    def _list_id_field(self, path: str) -> str | None:
        if path == "roles":
            return "name"
        if path == "workflow":
            return "id"
        if path == "quality_gates":
            return "name"
        if path == "tools.generated":
            return "name"
        return None

    def _infer_skill_type(self, field_path: str) -> SkillType | None:
        if field_path.startswith("workflow"):
            return "workflow_step"
        if field_path.startswith("quality_gates"):
            return "quality_gate"
        if field_path.startswith("tools"):
            return "tool"
        if field_path.startswith("roles") and ".instructions" in field_path:
            return "role_instruction"
        if field_path.startswith("roles"):
            return "role_instruction"
        if field_path.startswith("retry_policy"):
            return "retry_policy"
        if field_path.startswith("self_evaluation"):
            return "self_eval_rubric_fragment"
        if field_path.startswith("memory"):
            return "memory_schema"
        return None

    def _skill_name(self, mutation_type: str, field_path: str) -> str:
        safe_path = re.sub(r"[^A-Za-z0-9_.=-]+", "_", field_path).strip("_")
        return f"{mutation_type}:{safe_path}"

    def _skill_id(
        self,
        run_id: str,
        scenario_name: str,
        generation_number: int,
        mutation_type: str,
        field_path: str,
        fragment: Any,
    ) -> str:
        payload = json.dumps(
            {
                "run_id": run_id,
                "scenario_name": scenario_name,
                "generation": generation_number,
                "mutation_type": mutation_type,
                "field_path": field_path,
                "fragment": fragment,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"atom_{digest}"
