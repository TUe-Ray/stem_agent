from __future__ import annotations

import json
from pathlib import Path

from stem_agent.genome.models import Genome
from stem_agent.kernel.evaluator import EvaluationResult
from stem_agent.evolution.lineage import LineageLog
from stem_agent.evolution.visuals import VisualizationBuilder
from stem_agent.kernel.versioning import GenomeArchive


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
            "# stem_agent Evolution Report",
            "",
            "stem_agent is a universal differentiation mechanism. This run grew a specialized operating harness from scenario signals.",
            "",
            "## Before / After",
            f"- Baseline genome score: {baseline.promotion_score:.5f}",
            f"- Evolved genome score: {final.promotion_score:.5f}",
            f"- Frozen genome: `{frozen_path}`",
            f"- Promotion split policy: {final.split_policy}",
            "",
            "## Research Evaluation",
            *self._research_evaluation_table(run_dir, baseline, final),
            "",
            "## Fitness Vector",
            "| Metric | Baseline | Final |",
            "|---|---:|---:|",
            f"| Promotion score | {baseline.promotion_score:.5f} | {final.promotion_score:.5f} |",
            f"| Task quality | {baseline.task_quality:.5f} | {final.task_quality:.5f} |",
            f"| Train score | {baseline.train_score:.5f} | {final.train_score:.5f} |",
            f"| Validation score | {float(baseline.validation_score or 0.0):.5f} | {float(final.validation_score or 0.0):.5f} |",
            f"| Cost efficiency | {baseline.cost_efficiency:.5f} | {final.cost_efficiency:.5f} |",
            f"| Safety score | {baseline.safety_score:.5f} | {final.safety_score:.5f} |",
            f"| Stability score | {baseline.stability_score:.5f} | {final.stability_score:.5f} |",
            f"| Complexity penalty | {baseline.complexity_penalty:.5f} | {final.complexity_penalty:.5f} |",
            f"| Cost estimate | {baseline.cost_estimate:.5f} | {final.cost_estimate:.5f} |",
            "",
            "## Metric Delta",
            *self._metric_delta_table(baseline, final),
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
            "stem_agent does not start with a hand-written set of PM/Engineer/QA agents. It starts with a minimal Founder genome. Every new role, workflow step, quality gate, tool, or workspace artifact must appear as a mutation. Guardian evaluates the mutated harness and only promotes changes that improve fitness.",
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
                "## Genome Archive",
                GenomeArchive(run_dir=run_dir).to_markdown_table(),
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
        VisualizationBuilder().visualize(run_dir, update_report=True)
        return path

    def _research_evaluation_table(
        self,
        run_dir: Path,
        baseline: EvaluationResult,
        final: EvaluationResult,
    ) -> list[str]:
        rows = [("Internal promotion", baseline.promotion_score, final.promotion_score)]
        baseline_external = self._read_eval_score(
            run_dir / "generation_000" / "external_benchmark" / "eval_result.json"
        )
        final_external = self._read_eval_score(
            run_dir / "final_evaluation" / "external_benchmark" / "eval_result.json"
        )
        final_holdout = self._read_eval_score(run_dir / "final_holdout" / "eval_result.json")
        if baseline_external is not None or final_external is not None:
            rows.append(("External dev benchmark", baseline_external, final_external))
        if final_holdout is not None:
            rows.append(("Final holdout", None, final_holdout))

        table = ["| Split | Baseline | Final | Delta |", "|---|---:|---:|---:|"]
        for name, before, after in rows:
            delta = None if before is None or after is None else after - before
            table.append(
                f"| {name} | {self._fmt_optional(before)} | "
                f"{self._fmt_optional(after)} | {self._fmt_optional(delta)} |"
            )
        return table

    def _metric_delta_table(
        self,
        baseline: EvaluationResult,
        final: EvaluationResult,
    ) -> list[str]:
        preferred = [
            "requirement_coverage",
            "format_validity",
            "constraint_adherence",
            "actionability",
            "input_specificity",
            "artifact_presence",
            "self_review_usage",
            "workflow_completion",
            "quality_gate_usage",
            "generated_tool_usage",
            "cost_penalty",
            "complexity_penalty",
        ]
        keys = [key for key in preferred if key in baseline.metrics or key in final.metrics]
        if not keys:
            return ["No metric details were recorded."]
        table = ["| Metric | Baseline | Final | Delta |", "|---|---:|---:|---:|"]
        for key in keys:
            before = float(baseline.metrics.get(key, 0.0))
            after = float(final.metrics.get(key, 0.0))
            table.append(f"| {key} | {before:.5f} | {after:.5f} | {after - before:.5f} |")
        return table

    def _read_eval_score(self, path: Path) -> float | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("promotion_score", data.get("score", 0.0)))

    def _fmt_optional(self, value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.5f}"
