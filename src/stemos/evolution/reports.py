from __future__ import annotations

from pathlib import Path

from stemos.genome.models import Genome
from stemos.kernel.evaluator import EvaluationResult
from stemos.evolution.lineage import LineageLog


class ReportBuilder:
    def write_report(
        self,
        *,
        run_dir: Path,
        baseline: EvaluationResult,
        final: EvaluationResult,
        frozen_genome: Genome,
        lineage: LineageLog,
        frozen_path: Path,
    ) -> Path:
        promoted = [event for event in lineage.events if event["event"] == "mutation_promoted"]
        rejected = [
            event
            for event in lineage.events
            if event["event"] in {"mutation_rejected", "mutation_rolled_back"}
        ]
        content = [
            "# StemOS Evolution Report",
            "",
            "StemOS is a universal differentiation mechanism. This run grew a specialized operating harness from scenario signals.",
            "",
            "## Before / After",
            f"- Baseline genome score: {baseline.promotion_score:.4f}",
            f"- Evolved genome score: {final.promotion_score:.4f}",
            f"- Frozen genome: `{frozen_path}`",
            f"- Promotion split policy: {final.split_policy}",
            "",
            "## Promoted Mutations",
        ]
        if promoted:
            for event in promoted:
                content.append(
                    f"- {event['mutation_type']} on {event['target']}: {event.get('rationale', '')}"
                )
        else:
            content.append("- None")

        content.extend(["", "## Rejected Or Rolled Back Mutations"])
        if rejected:
            for event in rejected:
                reason = event.get("reason", "")
                content.append(f"- {event['event']} {event.get('mutation_type')} on {event.get('target')}: {reason}")
        else:
            content.append("- None")

        content.extend(
            [
                "",
                "## Frozen Harness Shape",
                f"- Roles: {', '.join(role.name for role in frozen_genome.roles)}",
                f"- Workflow steps: {', '.join(step.id for step in frozen_genome.workflow)}",
                f"- Environment artifacts: {', '.join(frozen_genome.environment.required_artifacts) or 'none'}",
                f"- Self-evaluation enabled: {frozen_genome.self_evaluation.get('enabled', False)}",
                "",
                "## Differentiation Story",
                lineage.differentiation_story(),
                "",
            ]
        )
        path = run_dir / "report.md"
        path.write_text("\n".join(content), encoding="utf-8")
        return path
