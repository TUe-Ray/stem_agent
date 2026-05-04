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
        baseline_genome: Genome,
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
        generated_tool_rejections = [
            event
            for event in lineage.events
            if event["event"] == "mutation_rejected"
            and event.get("mutation_type") == "create_tool"
        ]
        accepted_generated_tools = frozen_genome.tools.get("generated", []) or []
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
            "## Organism Shape",
            "",
            "### Initial organism",
            f"- Roles: {len(baseline_genome.roles)}",
            f"- Workflow steps: {len(baseline_genome.workflow)}",
            f"- Self-evaluation: {'enabled' if baseline_genome.self_evaluation.get('enabled') else 'disabled'}",
            f"- Quality gates: {len(baseline_genome.quality_gates)}",
            f"- Generated tools: {len(baseline_genome.tools.get('generated', []) or [])}",
            f"- Environment artifacts: {len(baseline_genome.environment.required_artifacts)}",
            "",
            "### Final organism",
            f"- Roles: {len(frozen_genome.roles)}",
            f"- Workflow steps: {len(frozen_genome.workflow)}",
            f"- Self-evaluation: {'enabled' if frozen_genome.self_evaluation.get('enabled') else 'disabled'}",
            f"- Quality gates: {len(frozen_genome.quality_gates)}",
            f"- Generated tools: {len(accepted_generated_tools)} accepted, {len(generated_tool_rejections)} rejected",
            f"- Environment artifacts: {len(frozen_genome.environment.required_artifacts)}",
            "",
            "## Why This Is Evolution, Not Subagent Orchestration",
            "StemOS does not start with a hand-written set of PM/Engineer/QA agents. It starts with a minimal Founder genome. Every new role, workflow step, quality gate, tool, or workspace artifact must appear as a mutation. Guardian evaluates the mutated harness and only promotes changes that improve fitness.",
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
