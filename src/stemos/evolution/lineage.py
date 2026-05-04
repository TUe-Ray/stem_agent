from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LineageLog:
    def __init__(self, run_dir: Path, *, reset: bool = False):
        self.run_dir = run_dir
        self.path = run_dir / "lineage.jsonl"
        self.events: list[dict[str, Any]] = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.events.append(json.loads(line))

    def record(self, event: str, **payload: Any) -> None:
        item = {"event": event, **payload}
        self.events.append(item)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    def record_evaluation(self, generation: int, score: float, summary: str) -> None:
        self.record("evaluation", generation=generation, score=round(score, 4), summary=summary)

    def record_mutation_plan(self, generation: int, summary: str, mutation_count: int) -> None:
        self.record(
            "mutation_plan",
            generation=generation,
            summary=summary,
            mutation_count=mutation_count,
        )

    def record_rejected_mutation(self, generation: int, mutation_type: str, target: str, reason: str) -> None:
        self.record(
            "mutation_rejected",
            generation=generation,
            mutation_type=mutation_type,
            target=target,
            reason=reason,
        )

    def record_promoted_mutation(
        self,
        generation: int,
        mutation_type: str,
        target: str,
        score_before: float,
        score_after: float,
        rationale: str,
    ) -> None:
        self.record(
            "mutation_promoted",
            generation=generation,
            mutation_type=mutation_type,
            target=target,
            score_before=round(score_before, 4),
            score_after=round(score_after, 4),
            rationale=rationale,
        )

    def record_rolled_back_mutation(
        self,
        generation: int,
        mutation_type: str,
        target: str,
        score_before: float,
        score_after: float,
        reason: str,
    ) -> None:
        self.record(
            "mutation_rolled_back",
            generation=generation,
            mutation_type=mutation_type,
            target=target,
            score_before=round(score_before, 4),
            score_after=round(score_after, 4),
            reason=reason,
        )

    def record_freeze(self, generation: int, best_score: float, reason: str) -> None:
        self.record(
            "freeze",
            generation=generation,
            best_score=round(best_score, 4),
            reason=reason,
        )

    def summary(self) -> str:
        parts: list[str] = []
        for event in self.events[-12:]:
            if event["event"].startswith("mutation"):
                parts.append(
                    f"{event['event']}:{event.get('mutation_type')}:{event.get('target')}"
                )
            elif event["event"] == "evaluation":
                parts.append(f"evaluation:{event.get('score')}")
            else:
                parts.append(event["event"])
        return "; ".join(parts)

    def differentiation_story(self) -> str:
        lines = [
            "StemOS differentiation story",
            "",
            "Scenario signals were interpreted by Nucleus, then converted into structured genome mutations.",
        ]
        for event in self.events:
            kind = event["event"]
            if kind == "evaluation":
                lines.append(
                    f"- Generation {event['generation']} evaluated at {event['score']}: {event['summary']}"
                )
            elif kind == "mutation_promoted":
                lines.append(
                    "- Guardian promoted "
                    f"{event['mutation_type']} on {event['target']} because score improved "
                    f"from {event['score_before']} to {event['score_after']}."
                )
            elif kind == "mutation_rejected":
                lines.append(
                    "- Guardian rejected "
                    f"{event['mutation_type']} on {event['target']}: {event['reason']}."
                )
            elif kind == "mutation_rolled_back":
                lines.append(
                    "- Guardian rolled back "
                    f"{event['mutation_type']} on {event['target']} because {event['reason']}."
                )
            elif kind == "freeze":
                lines.append(
                    f"- Evolution stopped at generation {event['generation']}: {event['reason']}"
                )
        return "\n".join(lines)
