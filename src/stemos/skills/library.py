from __future__ import annotations

import json
from pathlib import Path

from stemos.skills.schema import AtomicSkill, SkillReuseRecord, SkillType


class SkillLibrary:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else Path("skills") / "atoms.jsonl"

    def add_atom(self, atom: AtomicSkill) -> bool:
        if atom.score_lift < 0.02:
            return False
        atoms = self._load()
        replacement_index = next(
            (
                index
                for index, existing in enumerate(atoms)
                if existing.name == atom.name and existing.skill_type == atom.skill_type
            ),
            None,
        )
        if replacement_index is not None:
            if atoms[replacement_index].score_lift >= atom.score_lift:
                return False
            atoms[replacement_index] = atom
        else:
            atoms.append(atom)
        self._save(atoms)
        return True

    def query_atoms(
        self,
        domain_tags: list[str],
        *,
        skill_type: SkillType | None = None,
        current_run_id: str,
        current_scenario_name: str,
        top_k: int = 5,
    ) -> list[AtomicSkill]:
        _ = current_scenario_name
        query_tags = self._normalize_tags(domain_tags)
        ranked: list[tuple[int, float, AtomicSkill]] = []
        for atom in self._load():
            if skill_type is not None and atom.skill_type != skill_type:
                continue
            if self._evicted_for_run(atom, current_run_id):
                continue
            overlap = len(query_tags & self._normalize_tags(atom.domain_tags))
            if query_tags and overlap <= 0:
                continue
            ranked.append((overlap, atom.score_lift, atom))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [atom for _, _, atom in ranked[:top_k]]

    def open_probation(
        self,
        skill_id: str,
        *,
        run_id: str,
        scenario_name: str,
    ) -> None:
        atoms = self._load()
        changed = False
        for atom in atoms:
            if atom.id != skill_id:
                continue
            if any(record.run_id == run_id for record in atom.reuse_history):
                break
            atom.reuse_history.append(
                SkillReuseRecord(
                    run_id=run_id,
                    scenario_name=scenario_name,
                    generations_on_probation=0,
                    observed_score_lift=None,
                    evicted=False,
                )
            )
            changed = True
            break
        if changed:
            self._save(atoms)

    def tick_probation(self, *, run_id: str) -> list[str]:
        atoms = self._load()
        evicted: list[str] = []
        changed = False
        for atom in atoms:
            for record in atom.reuse_history:
                if record.run_id != run_id or record.evicted:
                    continue
                record.generations_on_probation += 1
                if (
                    record.generations_on_probation >= 2
                    and (record.observed_score_lift is None or record.observed_score_lift < 0.02)
                ):
                    record.evicted = True
                    evicted.append(atom.id)
                changed = True
        if changed:
            self._save(atoms)
        return evicted

    def record_observed_lift(
        self,
        *,
        skill_id: str,
        run_id: str,
        observed_score_lift: float,
    ) -> None:
        atoms = self._load()
        changed = False
        for atom in atoms:
            if atom.id != skill_id:
                continue
            for record in atom.reuse_history:
                if record.run_id == run_id and not record.evicted:
                    record.observed_score_lift = observed_score_lift
                    changed = True
        if changed:
            self._save(atoms)

    def saturation_score(self, domain_tags: list[str]) -> float:
        atoms = self.query_atoms(
            domain_tags,
            current_run_id="",
            current_scenario_name="",
            top_k=5,
        )
        if not atoms:
            return 0.0
        saturated = 0
        for atom in atoms:
            strong_origin = atom.score_lift > 0.05
            reused_successfully = any(
                not record.evicted
                and record.observed_score_lift is not None
                and record.observed_score_lift > 0
                for record in atom.reuse_history
            )
            if strong_origin or reused_successfully:
                saturated += 1
        return saturated / len(atoms)

    def _load(self) -> list[AtomicSkill]:
        if not self.path.exists():
            return []
        atoms: list[AtomicSkill] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                atoms.append(AtomicSkill.model_validate(json.loads(line)))
        return atoms

    def _save(self, atoms: list[AtomicSkill]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for atom in atoms:
                handle.write(json.dumps(atom.model_dump(mode="json"), sort_keys=True) + "\n")

    def _evicted_for_run(self, atom: AtomicSkill, run_id: str) -> bool:
        if not run_id:
            return False
        return any(record.run_id == run_id and record.evicted for record in atom.reuse_history)

    def _normalize_tags(self, tags: list[str]) -> set[str]:
        return {tag.strip().lower() for tag in tags if tag.strip()}
