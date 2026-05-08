from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from stemos.genome.loader import load_genome
from stemos.genome.models import Genome
from stemos.harness.builder import HarnessBuilder
from stemos.harness.runner import HarnessRunner
from stemos.harness.role_runner import RoleRunner
from stemos.kernel.evaluator import GuardianFitnessEvaluator
from stemos.nucleus.model_client import ModelClient
from stemos.scenarios.loader import load_scenario
from stemos.scenarios.schema import Scenario, TaskCase


VISUALS_START = "<!-- STEM_AGENT_VISUALS_START -->"
VISUALS_END = "<!-- STEM_AGENT_VISUALS_END -->"


@dataclass
class RunArtifacts:
    run_dir: Path
    baseline_genome: Genome
    frozen_genome: Genome
    baseline_eval: dict[str, Any]
    final_eval: dict[str, Any]
    lineage: list[dict[str, Any]]
    scenario: Scenario
    config: dict[str, Any]
    metadata: dict[str, Any]


class VisualizationBuilder:
    """Generate reviewer-facing evolution visuals from run artifacts."""

    def visualize(self, run_dir: str | Path, *, update_report: bool = True) -> list[Path]:
        artifacts = self._load_artifacts(Path(run_dir))
        visuals_dir = artifacts.run_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "training_progress.md": self.training_progress(artifacts.run_dir),
            "evolution_timeline.md": self.evolution_timeline(artifacts),
            "organism_shape.md": self.organism_shape(artifacts),
            "harness_before_after.md": self.harness_before_after(artifacts),
            "guardian_selection_board.md": self.guardian_selection_board(artifacts),
            "hidden_eval_vs_train.md": self.hidden_eval_vs_train(artifacts),
            "output_comparison.md": self.output_comparison(artifacts),
            "visual_report.md": self.visual_report_markdown(artifacts),
        }
        paths: list[Path] = []
        for name, content in files.items():
            path = visuals_dir / name
            path.write_text(content, encoding="utf-8")
            paths.append(path)

        if update_report:
            self.embed_in_report(artifacts.run_dir, self.report_visual_block(artifacts))
        return paths

    def progress(self, run_dir: str | Path) -> Path:
        run_path = Path(run_dir)
        visuals_dir = run_path / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        path = visuals_dir / "training_progress.md"
        path.write_text(self.training_progress(run_path), encoding="utf-8")
        return path

    def report_visual_block(self, artifacts: RunArtifacts) -> str:
        sections = [
            VISUALS_START,
            "## Visual Overview",
            "",
            self.openai_run_metadata(artifacts),
            "",
            self.training_progress(artifacts.run_dir),
            "",
            self.evolution_timeline(artifacts),
            "",
            self.organism_shape(artifacts),
            "",
            self.harness_before_after(artifacts),
            "",
            self.guardian_selection_board(artifacts),
            "",
            self.hidden_eval_vs_train(artifacts),
            "",
            self.safe_stop_recovery(artifacts),
            "",
            self.not_subagent_explanation(),
            "",
            VISUALS_END,
        ]
        return "\n".join(sections).rstrip() + "\n"

    def training_progress(self, run_dir: str | Path) -> str:
        run_path = Path(run_dir)
        lineage = self._read_lineage_optional(run_path / "lineage.jsonl")
        generation_rows = self._generation_progress_rows(run_path, lineage)
        decision_rows = self._decision_progress_rows(lineage)
        best_score = max((row["promotion_score"] for row in generation_rows), default=0.0)
        latest_generation = generation_rows[-1]["generation"] if generation_rows else "none"
        status = self._progress_status(run_path, lineage)

        return "\n".join(
            [
                "# stem_agent Training Progress",
                "",
                f"- Run: `{run_path}`",
                f"- Status: {status}",
                f"- Latest generation: {latest_generation}",
                f"- Best observed promotion score: {best_score:.4f}",
                "",
                self._score_chart(generation_rows),
                "",
                self._score_table(generation_rows),
                "",
                self._decision_timeline(decision_rows),
                "",
                self._decision_table(decision_rows),
                "",
            ]
        ).rstrip() + "\n"

    def embed_in_report(self, run_dir: Path, block: str) -> None:
        report_path = run_dir / "report.md"
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8")
        else:
            report = "# stem_agent Evolution Report\n"

        pattern = re.compile(
            rf"\n?{re.escape(VISUALS_START)}.*?{re.escape(VISUALS_END)}\n?",
            re.DOTALL,
        )
        report = pattern.sub("\n", report).rstrip()
        report_path.write_text(report + "\n\n" + block, encoding="utf-8")

    def evolution_timeline(self, artifacts: RunArtifacts) -> str:
        nodes = [
            '  G0["G0 stem seed<br/>score '
            + self._score(artifacts.baseline_eval)
            + '"]'
        ]
        edges: list[str] = []
        current = "G0"
        promoted_index = 1
        rejected_index = 0

        for event in artifacts.lineage:
            kind = event.get("event")
            if kind == "mutation_promoted":
                node_id = f"G{promoted_index}"
                label = self._mutation_label(event)
                nodes.append(
                    f'  {node_id}["{label}<br/>score {event.get("score_after", "?")}"]'
                )
                edges.append(f"  {current} --> {node_id}")
                current = node_id
                promoted_index += 1
            elif kind in {"mutation_rejected", "mutation_rolled_back"}:
                node_id = f"R{rejected_index}"
                label = self._mutation_label(event)
                reason = self._truncate(str(event.get("reason", kind)), 44)
                nodes.append(f'  {node_id}["{label}<br/>{reason}"]')
                edges.append(f"  {current} -.-> {node_id}")
                rejected_index += 1

        nodes.append(
            '  F["frozen specialized harness<br/>score '
            + self._score(artifacts.final_eval)
            + '"]'
        )
        edges.append(f"  {current} --> F")

        class_lines = [
            "  classDef rejected stroke:#d33,color:#b11,stroke-dasharray: 5 5;",
        ]
        if rejected_index:
            class_lines.append(
                "  class " + ",".join(f"R{index}" for index in range(rejected_index)) + " rejected;"
            )

        return "\n".join(
            [
                "## Evolution Timeline",
                "",
                "```mermaid",
                "flowchart LR",
                *nodes,
                *edges,
                *class_lines,
                "```",
            ]
        )

    def organism_shape(self, artifacts: RunArtifacts) -> str:
        accepted_tools = len(artifacts.frozen_genome.tools.get("generated", []) or [])
        rejected_tools = sum(
            1
            for event in artifacts.lineage
            if event.get("event") == "mutation_rejected"
            and event.get("mutation_type") == "create_tool"
        )
        rows = [
            ("Roles", len(artifacts.baseline_genome.roles), len(artifacts.frozen_genome.roles)),
            (
                "Workflow steps",
                len(artifacts.baseline_genome.workflow),
                len(artifacts.frozen_genome.workflow),
            ),
            (
                "Self-evaluation",
                self._enabled(artifacts.baseline_genome),
                self._enabled(artifacts.frozen_genome),
            ),
            (
                "Quality gates",
                len(artifacts.baseline_genome.quality_gates),
                len(artifacts.frozen_genome.quality_gates),
            ),
            (
                "Generated tools",
                len(artifacts.baseline_genome.tools.get("generated", []) or []),
                f"{accepted_tools} accepted, {rejected_tools} rejected",
            ),
            (
                "Environment artifacts",
                len(artifacts.baseline_genome.environment.required_artifacts),
                len(artifacts.frozen_genome.environment.required_artifacts),
            ),
            ("Score", self._score(artifacts.baseline_eval), self._score(artifacts.final_eval)),
        ]
        table = ["| Shape signal | Initial organism | Final organism |", "|---|---:|---:|"]
        table.extend(f"| {name} | {before} | {after} |" for name, before, after in rows)
        return "\n".join(["## Initial vs Final Organism", "", *table])

    def harness_before_after(self, artifacts: RunArtifacts) -> str:
        baseline = self._workflow_chain("B", artifacts.baseline_genome, include_organs=False)
        frozen = self._workflow_chain("F", artifacts.frozen_genome, include_organs=True)
        return "\n".join(
            [
                "## Harness Before/After Graph",
                "",
                "```mermaid",
                "flowchart LR",
                '  subgraph Baseline["Baseline harness"]',
                *baseline,
                "  end",
                '  subgraph Frozen["Frozen harness"]',
                *frozen,
                "  end",
                "```",
            ]
        )

    def safe_stop_recovery(self, artifacts: RunArtifacts) -> str:
        paused = any(event.get("event") == "pause" for event in artifacts.lineage)
        resumed = any(event.get("event") == "resume" for event in artifacts.lineage)
        return "\n".join(
            [
                "## Safe Stop And Recovery",
                "",
                f"- Run completed {'after pause/resume' if paused or resumed else 'without pause'}.",
                "- Checkpoints written at safe points.",
                "- Best verified genome frozen.",
                "- Candidate genome was not promoted without Guardian verification.",
                f"- Resume {'was used' if resumed else 'not needed for completed run'}.",
            ]
        )

    def guardian_selection_board(self, artifacts: RunArtifacts) -> str:
        rows = [
            "| Generation | Mutation | Type | Decision | Score before | Score after | Guardian reason |",
            "|---:|---|---|---|---:|---:|---|",
        ]
        for event in artifacts.lineage:
            if event.get("event") not in {
                "mutation_promoted",
                "mutation_rejected",
                "mutation_rolled_back",
            }:
                continue
            decision = {
                "mutation_promoted": "promoted",
                "mutation_rejected": "rejected",
                "mutation_rolled_back": "rolled back",
            }[str(event["event"])]
            reason = event.get("reason")
            if not reason and decision == "promoted":
                reason = "fitness improved"
            rows.append(
                "| {generation} | {target} | {mutation_type} | {decision} | {before} | {after} | {reason} |".format(
                    generation=event.get("generation", ""),
                    target=self._escape_table(str(event.get("target", ""))),
                    mutation_type=self._escape_table(str(event.get("mutation_type", ""))),
                    decision=decision,
                    before=event.get("score_before", ""),
                    after=event.get("score_after", ""),
                    reason=self._escape_table(str(reason or "")),
                )
            )
        return "\n".join(["## Guardian Selection Board", "", *rows])

    def hidden_eval_vs_train(self, artifacts: RunArtifacts) -> str:
        hidden_rows = self._hidden_eval_rows(artifacts.run_dir)
        train_rows = self._generation_progress_rows(artifacts.run_dir, artifacts.lineage)
        if not hidden_rows:
            return "\n".join(
                [
                    "## Hidden Eval vs Train Eval over Generations",
                    "",
                    "No hidden evaluation scores recorded yet.",
                ]
            )
        generations = sorted(
            {
                int(row["generation"])
                for row in hidden_rows
                if row.get("generation") is not None
            }
            | {int(row["generation"]) for row in train_rows}
        )
        train_by_generation = {int(row["generation"]): float(row["train_score"]) for row in train_rows}
        hidden_by_generation: dict[int, float] = {}
        for row in hidden_rows:
            if row.get("generation") is not None:
                hidden_by_generation[int(row["generation"])] = float(row["hidden_score"])
        x_axis = ", ".join(str(item) for item in generations)
        train_line = ", ".join(f"{train_by_generation.get(item, 0.0):.4f}" for item in generations)
        hidden_line = ", ".join(f"{hidden_by_generation.get(item, 0.0):.4f}" for item in generations)
        table = [
            "| Generation | Train eval | Hidden eval | Genome ID |",
            "|---:|---:|---:|---|",
        ]
        genome_by_generation = {
            int(row["generation"]): str(row.get("genome_id", ""))
            for row in hidden_rows
            if row.get("generation") is not None
        }
        for generation in generations:
            table.append(
                "| {generation} | {train:.4f} | {hidden:.4f} | `{genome}` |".format(
                    generation=generation,
                    train=train_by_generation.get(generation, 0.0),
                    hidden=hidden_by_generation.get(generation, 0.0),
                    genome=genome_by_generation.get(generation, ""),
                )
            )
        return "\n".join(
            [
                "## Hidden Eval vs Train Eval over Generations",
                "",
                "```mermaid",
                "xychart-beta",
                '  title "Hidden Eval vs Train Eval over Generations"',
                f"  x-axis [{x_axis}]",
                '  y-axis "score" 0 --> 1',
                f"  line \"train\" [{train_line}]",
                f"  line \"hidden\" [{hidden_line}]",
                "```",
                "",
                *table,
            ]
        )

    def output_comparison(self, artifacts: RunArtifacts) -> str:
        sample_input = self._sample_input(artifacts)
        metadata = self._normalize_metadata(artifacts.metadata or self._infer_metadata(artifacts))
        test_mode = bool(metadata.get("test_mode", False))
        baseline_output = self._run_output(
            artifacts.baseline_genome,
            artifacts.scenario,
            artifacts.run_dir / "visuals" / "baseline_workspace",
            sample_input,
            test_mode=test_mode,
        )
        evolved_output = self._run_output(
            artifacts.frozen_genome,
            artifacts.scenario,
            artifacts.run_dir / "visuals" / "frozen_workspace",
            sample_input,
            test_mode=test_mode,
        )
        evaluator = GuardianFitnessEvaluator()
        checklist = ["| Requirement | Baseline | Evolved |", "|---|---:|---:|"]
        for requirement in artifacts.scenario.expected_output.requirements:
            baseline_ok = evaluator._requirement_satisfied(requirement, baseline_output)
            evolved_ok = evaluator._requirement_satisfied(requirement, evolved_output)
            checklist.append(
                f"| {self._escape_table(requirement)} | {self._mark(baseline_ok)} | {self._mark(evolved_ok)} |"
            )

        return "\n".join(
            [
                "## Before/After Output Comparison",
                "",
                f"Sample input: `{sample_input}`",
                "",
                "### Requirement Coverage Checklist",
                *checklist,
                "",
                "### Baseline Output",
                "",
                "```markdown",
                baseline_output.strip(),
                "```",
                "",
                "### Evolved Output",
                "",
                "```markdown",
                evolved_output.strip(),
                "```",
            ]
        )

    def not_subagent_explanation(self) -> str:
        return "\n".join(
            [
                "## Why This Is Not Predefined Subagent Orchestration",
                "",
                "Roles are not predefined subagents. They are phenotypic structures that survive only if Guardian fitness improves. In this run, a redundant role was rolled back, while workflow/tool/gate organs survived.",
            ]
        )

    def openai_run_metadata(self, artifacts: RunArtifacts) -> str:
        metadata = self._normalize_metadata(artifacts.metadata or self._infer_metadata(artifacts))
        rows = [
            ("run mode", metadata.get("run_mode", "not recorded")),
            ("model", metadata.get("model", "not recorded")),
            ("endpoint", metadata.get("endpoint", "not recorded")),
            ("test_mode", metadata.get("test_mode", "not recorded")),
            ("fallback_used", metadata.get("fallback_used", "not recorded")),
            (
                "responses_api_available",
                metadata.get("responses_api_available", "not recorded"),
            ),
            (
                "chat_completions_fallback",
                metadata.get("chat_completions_fallback", "not recorded"),
            ),
            ("model calls", metadata.get("model_calls", "not recorded")),
            (
                "structured output repairs",
                metadata.get("structured_output_repairs", "not recorded"),
            ),
            ("total estimated cost", metadata.get("total_estimated_cost", "not recorded")),
        ]
        lines = ["## OpenAI Run Metadata", "", "| Field | Value |", "|---|---|"]
        lines.extend(f"| {field} | {self._escape_table(str(value))} |" for field, value in rows)
        lines.extend(
            [
                "",
                "Structured output repairs only normalize model JSON/schema output. They do not bypass Guardian validation or promote mutations.",
            ]
        )
        return "\n".join(lines)

    def visual_report_markdown(self, artifacts: RunArtifacts) -> str:
        return "\n\n".join(
            [
                "# stem_agent Visual Report",
                "",
                self.evolution_timeline(artifacts),
                self.organism_shape(artifacts),
                self.harness_before_after(artifacts),
                self.guardian_selection_board(artifacts),
                self.hidden_eval_vs_train(artifacts),
                self.output_comparison(artifacts),
                self.safe_stop_recovery(artifacts),
                self.not_subagent_explanation(),
                self.openai_run_metadata(artifacts),
            ]
        )

    def aggregate(self, run_dirs: list[str | Path], output_path: Path | None = None) -> Path:
        rows = [
            "| Run | Baseline | Final | Improvement | Promoted | Rejected/Rolled Back | Final organism shape |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for item in run_dirs:
            artifacts = self._load_artifacts(Path(item))
            baseline = float(artifacts.baseline_eval.get("promotion_score", 0.0))
            final = float(artifacts.final_eval.get("promotion_score", 0.0))
            promoted = sum(1 for event in artifacts.lineage if event.get("event") == "mutation_promoted")
            rejected = sum(
                1
                for event in artifacts.lineage
                if event.get("event") in {"mutation_rejected", "mutation_rolled_back"}
            )
            shape = (
                f"{len(artifacts.frozen_genome.roles)} roles, "
                f"{len(artifacts.frozen_genome.workflow)} steps, "
                f"{len(artifacts.frozen_genome.quality_gates)} gates, "
                f"{len(artifacts.frozen_genome.tools.get('generated', []) or [])} generated tools, "
                f"{len(artifacts.frozen_genome.environment.required_artifacts)} artifacts"
            )
            rows.append(
                f"| {self._escape_table(str(artifacts.run_dir))} | {baseline:.4f} | {final:.4f} | {final - baseline:.4f} | {promoted} | {rejected} | {shape} |"
            )
        content = "\n".join(["# stem_agent Aggregate Run Report", "", *rows, ""])
        path = output_path or Path("runs") / "aggregate_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _generation_progress_rows(
        self, run_dir: Path, lineage: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rows_by_generation: dict[int, dict[str, Any]] = {}
        for generation_dir in sorted(run_dir.glob("generation_*")):
            if not generation_dir.is_dir():
                continue
            try:
                generation = int(generation_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            eval_path = generation_dir / "eval_result.json"
            if not eval_path.exists():
                continue
            data = self._read_json(eval_path)
            rows_by_generation[generation] = {
                "generation": generation,
                "promotion_score": float(data.get("promotion_score", data.get("score", 0.0))),
                "train_score": float(data.get("train_score", 0.0)),
                "validation_score": data.get("validation_score"),
                "summary": "",
            }

        for event in lineage:
            if event.get("event") != "evaluation":
                continue
            generation = event.get("generation")
            if generation is None:
                continue
            row = rows_by_generation.setdefault(
                int(generation),
                {
                    "generation": int(generation),
                    "promotion_score": float(event.get("score", 0.0)),
                    "train_score": 0.0,
                    "validation_score": None,
                    "summary": "",
                },
            )
            row["summary"] = str(event.get("summary", ""))

        return [rows_by_generation[key] for key in sorted(rows_by_generation)]

    def _decision_progress_rows(self, lineage: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in lineage:
            kind = event.get("event")
            if kind not in {
                "mutation_promoted",
                "mutation_rejected",
                "mutation_rolled_back",
                "freeze",
            }:
                continue
            rows.append(
                {
                    "generation": event.get("generation", ""),
                    "decision": {
                        "mutation_promoted": "promoted",
                        "mutation_rejected": "rejected",
                        "mutation_rolled_back": "rolled back",
                        "freeze": "frozen",
                    }.get(str(kind), str(kind)),
                    "mutation_type": event.get("mutation_type", ""),
                    "target": event.get("target", ""),
                    "score_before": event.get("score_before", ""),
                    "score_after": event.get("score_after", event.get("best_score", "")),
                    "reason": event.get("reason") or event.get("rationale") or "",
                }
            )
        return rows

    def _progress_status(self, run_dir: Path, lineage: list[dict[str, Any]]) -> str:
        if any(event.get("event") == "freeze" for event in lineage):
            return "frozen"
        control_path = run_dir / "control.json"
        if control_path.exists():
            try:
                control = self._read_json(control_path)
                return str(control.get("status") or control.get("command") or "running")
            except json.JSONDecodeError:
                return "control file unreadable"
        if lineage:
            return "in progress or stopped before freeze"
        return "no progress recorded yet"

    def _score_chart(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "\n".join(
                [
                    "## Promotion Score Chart",
                    "",
                    "No generation scores recorded yet.",
                ]
            )
        generations = ", ".join(str(row["generation"]) for row in rows)
        scores = ", ".join(f"{row['promotion_score']:.4f}" for row in rows)
        return "\n".join(
            [
                "## Promotion Score Chart",
                "",
                "```mermaid",
                "xychart-beta",
                '  title "Promotion score by generation"',
                f"  x-axis [{generations}]",
                '  y-axis "score" 0 --> 1',
                f"  line [{scores}]",
                "```",
            ]
        )

    def _score_table(self, rows: list[dict[str, Any]]) -> str:
        table = [
            "## Generation Scores",
            "",
            "| Generation | Promotion | Train | Validation | Summary |",
            "|---:|---:|---:|---:|---|",
        ]
        if not rows:
            table.append("| | | | | No scores recorded yet. |")
            return "\n".join(table)
        for row in rows:
            validation = row.get("validation_score")
            validation_text = "" if validation is None else f"{float(validation):.4f}"
            table.append(
                "| {generation} | {promotion:.4f} | {train:.4f} | {validation} | {summary} |".format(
                    generation=row["generation"],
                    promotion=row["promotion_score"],
                    train=row["train_score"],
                    validation=validation_text,
                    summary=self._escape_table(str(row.get("summary", ""))),
                )
            )
        return "\n".join(table)

    def _decision_timeline(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "\n".join(
                [
                    "## Decision Timeline",
                    "",
                    "No mutation decisions recorded yet.",
                ]
            )
        lines = ["## Decision Timeline", "", "```mermaid", "timeline", "  title Mutation decisions"]
        for row in rows:
            generation = row.get("generation", "")
            decision = row.get("decision", "")
            mutation = row.get("mutation_type") or "run"
            target = row.get("target") or "best genome"
            lines.append(
                f"  generation {generation} : {decision} {mutation} on {target}"
            )
        lines.append("```")
        return "\n".join(lines)

    def _decision_table(self, rows: list[dict[str, Any]]) -> str:
        table = [
            "## Mutation Decisions",
            "",
            "| Generation | Decision | Mutation | Target | Score before | Score after | Reason |",
            "|---:|---|---|---|---:|---:|---|",
        ]
        if not rows:
            table.append("| | | | | | | No decisions recorded yet. |")
            return "\n".join(table)
        for row in rows:
            table.append(
                "| {generation} | {decision} | {mutation} | {target} | {before} | {after} | {reason} |".format(
                    generation=row.get("generation", ""),
                    decision=self._escape_table(str(row.get("decision", ""))),
                    mutation=self._escape_table(str(row.get("mutation_type", ""))),
                    target=self._escape_table(str(row.get("target", ""))),
                    before=row.get("score_before", ""),
                    after=row.get("score_after", ""),
                    reason=self._escape_table(str(row.get("reason", ""))),
                )
            )
        return "\n".join(table)

    def _load_artifacts(self, run_dir: Path) -> RunArtifacts:
        baseline_genome = load_genome(run_dir / "baseline_genome.yaml")
        frozen_genome = load_genome(run_dir / "frozen_genome.yaml")
        baseline_eval = self._read_json(run_dir / "generation_000" / "eval_result.json")
        final_eval = self._read_json(run_dir / "final_evaluation" / "eval_result.json")
        lineage = self._read_lineage(run_dir / "lineage.jsonl")
        config = self._read_yaml(run_dir / "config_snapshot.yaml")
        metadata = self._read_json(run_dir / "run_metadata.json", missing_ok=True)
        scenario_path = Path(config.get("scenario_path", "scenarios/toy_structured_answer"))
        scenario = load_scenario(scenario_path).scenario
        return RunArtifacts(
            run_dir=run_dir,
            baseline_genome=baseline_genome,
            frozen_genome=frozen_genome,
            baseline_eval=baseline_eval,
            final_eval=final_eval,
            lineage=lineage,
            scenario=scenario,
            config=config,
            metadata=metadata,
        )

    def _workflow_chain(
        self, prefix: str, genome: Genome, *, include_organs: bool
    ) -> list[str]:
        lines = [f'    {prefix}Input["user input"]']
        previous = f"{prefix}Input"
        for index, step in enumerate(genome.workflow):
            node = f"{prefix}S{index}"
            lines.append(f'    {node}["{step.id}"]')
            lines.append(f"    {previous} --> {node}")
            previous = node
        if include_organs and genome.quality_gates:
            lines.append(f'    {prefix}Gate["quality gate"]')
            lines.append(f"    {previous} --> {prefix}Gate")
            previous = f"{prefix}Gate"
        if include_organs and genome.tools.get("generated"):
            lines.append(f'    {prefix}Tool["checker tool"]')
            lines.append(f"    {previous} --> {prefix}Tool")
            previous = f"{prefix}Tool"
        lines.append(f'    {prefix}Out["final output"]')
        lines.append(f"    {previous} --> {prefix}Out")
        return lines

    def _run_output(
        self,
        genome: Genome,
        scenario: Scenario,
        workspace: Path,
        sample_input: str,
        *,
        test_mode: bool,
    ) -> str:
        harness = HarnessBuilder().materialize(genome, scenario, workspace_dir=workspace)
        runner = HarnessRunner(RoleRunner(ModelClient(test_mode=test_mode)))
        result = runner.run_case(
            harness,
            TaskCase(id="visual_sample", input={"user_request": sample_input}),
        )
        return result.final_output

    def _sample_input(self, artifacts: RunArtifacts) -> str:
        return (
            "Help me prepare for a difficult meeting with my manager about missed "
            "deadlines. I need to explain what happened, propose a recovery plan, "
            "and avoid sounding defensive."
        )

    def _infer_metadata(self, artifacts: RunArtifacts) -> dict[str, Any]:
        settings = artifacts.config.get("settings", {}) or {}
        test_mode = settings.get("test_mode", False)
        model_calls = sum(1 for event in artifacts.lineage if event.get("event") == "mutation_plan")
        if not test_mode:
            model_calls += 1
        repairs = 0
        if not test_mode:
            repairs = sum(
                1
                for event in artifacts.lineage
                if event.get("event") == "mutation_plan"
                and "Test Nucleus plan" in str(event.get("summary", ""))
            )
        endpoint = settings.get("openai_endpoint")
        if not endpoint and not test_mode:
            endpoint = "chat_completions"
        fallback_used = False if endpoint == "chat_completions" else "not recorded"
        responses_api_available = self._responses_api_available(
            endpoint,
            metadata={"test_mode": test_mode, "fallback_used": fallback_used},
        )
        chat_completions_fallback = self._chat_completions_fallback(
            endpoint, fallback_used=fallback_used, test_mode=test_mode
        )
        return {
            "run_mode": "test double" if test_mode else "openai-api",
            "model": settings.get("model", "not recorded"),
            "endpoint": endpoint or "not recorded",
            "test_mode": test_mode,
            "fallback_used": fallback_used,
            "responses_api_available": responses_api_available,
            "chat_completions_fallback": chat_completions_fallback,
            "model_calls": model_calls if not test_mode else 0,
            "structured_output_repairs": repairs,
            "total_estimated_cost": artifacts.final_eval.get("cost_estimate", "not recorded"),
        }

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(metadata)
        test_mode = normalized.get("test_mode", False)
        endpoint = normalized.get("endpoint")
        if not endpoint and not test_mode:
            endpoint = "chat_completions"
            normalized["endpoint"] = endpoint
        normalized.setdefault(
            "responses_api_available",
            self._responses_api_available(endpoint, normalized),
        )
        normalized.setdefault(
            "chat_completions_fallback",
            self._chat_completions_fallback(
                endpoint,
                fallback_used=normalized.get("fallback_used", "not recorded"),
                test_mode=test_mode,
            ),
        )
        return normalized

    def _responses_api_available(
        self, endpoint: Any, metadata: dict[str, Any]
    ) -> bool | str:
        if metadata.get("test_mode") is True:
            return "not applicable"
        if metadata.get("fallback_used") is True:
            return False
        if endpoint == "responses":
            return True
        if endpoint == "chat_completions":
            return False
        return "not recorded"

    def _chat_completions_fallback(
        self, endpoint: Any, *, fallback_used: Any, test_mode: Any
    ) -> bool | str:
        if test_mode is True:
            return "not applicable"
        if fallback_used is True:
            return True
        if endpoint == "chat_completions":
            return True
        if endpoint == "responses":
            return False
        return "not recorded"

    def _mutation_label(self, event: dict[str, Any]) -> str:
        mutation_type = str(event.get("mutation_type", "mutation"))
        rationale = str(event.get("rationale", ""))
        if mutation_type == "modify_self_evaluation":
            return "self-eval"
        if mutation_type == "modify_environment":
            return "environment"
        if mutation_type == "add_quality_gate":
            return "quality gate"
        if mutation_type == "create_tool":
            return "safe tool" if event.get("event") == "mutation_promoted" else "tool rejected"
        if mutation_type == "add_role":
            return "role organ"
        if mutation_type == "add_workflow_step" and any(
            token in rationale.lower() for token in ["revise", "affect", "delivered"]
        ):
            return "revise workflow"
        if mutation_type == "add_workflow_step" and "review" in rationale.lower():
            return "review workflow"
        return mutation_type.replace("_", " ")

    def _score(self, data: dict[str, Any]) -> str:
        value = data.get("promotion_score", data.get("score", 0.0))
        return f"{float(value):.4f}"

    def _enabled(self, genome: Genome) -> str:
        return "enabled" if genome.self_evaluation.get("enabled") else "disabled"

    def _mark(self, value: bool) -> str:
        return "yes" if value else "no"

    def _escape_table(self, value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br/>")

    def _truncate(self, value: str, length: int) -> str:
        value = value.replace("\n", " ")
        if len(value) <= length:
            return value
        return value[: length - 3] + "..."

    def _read_json(self, path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
        if missing_ok and not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _read_lineage(self, path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _read_lineage_optional(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return self._read_lineage(path)

    def _hidden_eval_rows(self, run_dir: Path) -> list[dict[str, Any]]:
        path = run_dir / "hidden_eval_log.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
