from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from stemos.config import Settings, load_settings
from stemos.evolution.lineage import LineageLog
from stemos.evolution.reports import ReportBuilder
from stemos.genome.loader import load_default_genome
from stemos.genome.models import Genome
from stemos.genome.serializer import save_genome
from stemos.harness.builder import HarnessBuilder
from stemos.harness.runner import HarnessRunResult, HarnessRunner
from stemos.kernel.budget import BudgetTracker
from stemos.kernel.evaluator import EvaluationResult
from stemos.kernel.guardian import Guardian
from stemos.nucleus.commander import NucleusCommander
from stemos.nucleus.failure_analyzer import FailureAnalyzer
from stemos.nucleus.model_client import ModelClient
from stemos.nucleus.mutation_planner import MutationPlanner
from stemos.nucleus.scenario_interpreter import ScenarioInterpreter
from stemos.scenarios.loader import ScenarioBundle, load_scenario


class EvolutionRunResult(BaseModel):
    run_dir: Path
    baseline_score: float
    final_score: float
    frozen_genome_path: Path
    report_path: Path


class EvolutionLoop:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        guardian: Guardian | None = None,
        harness_builder: HarnessBuilder | None = None,
        harness_runner: HarnessRunner | None = None,
        runs_root: str | Path = "runs",
    ):
        self.settings = settings or load_settings()
        model_client = ModelClient(model=self.settings.model, offline=self.settings.offline_mode)
        self.nucleus = NucleusCommander(
            scenario_interpreter=ScenarioInterpreter(model_client),
            failure_analyzer=FailureAnalyzer(),
            mutation_planner=MutationPlanner(model_client),
        )
        self.guardian = guardian or Guardian()
        self.harness_builder = harness_builder or HarnessBuilder()
        self.harness_runner = harness_runner or HarnessRunner()
        self.runs_root = Path(runs_root)

    def evolve(self, scenario_path: str | Path, run_id: str) -> EvolutionRunResult:
        bundle = load_scenario(scenario_path)
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        lineage = LineageLog(run_dir, reset=True)
        budget = BudgetTracker(max_cost_usd=bundle.scenario.evolution.max_cost_usd)
        self._write_config_snapshot(run_dir, bundle)

        diagnosis = self.nucleus.diagnose(bundle.scenario)
        genome = self.nucleus.attach_diagnosis(load_default_genome(), diagnosis, bundle.scenario)
        save_genome(genome, run_dir / "baseline_genome.yaml")

        best_genome = genome
        best_result: EvaluationResult | None = None
        baseline_result: EvaluationResult | None = None
        patience_left = bundle.scenario.evolution.patience
        stop_reason = "maximum generations reached"
        history: list[EvaluationResult] = []

        for generation in range(bundle.scenario.evolution.max_generations):
            generation_dir = run_dir / f"generation_{generation:03d}"
            generation_dir.mkdir(parents=True, exist_ok=True)
            save_genome(genome, generation_dir / "genome.yaml")

            current_result = self._run_and_evaluate(genome, bundle, generation_dir, "current")
            history.append(current_result)
            budget.record(current_result.cost_estimate)
            self._write_json(generation_dir / "eval_result.json", current_result.model_dump(mode="json"))

            if baseline_result is None:
                baseline_result = current_result

            summary = self._evaluation_summary(current_result)
            lineage.record_evaluation(generation, current_result.promotion_score, summary)

            if best_result is None or self.guardian.should_promote(
                best_result.promotion_score,
                current_result.promotion_score,
                bundle.scenario.evolution.min_delta,
            ):
                best_result = current_result
                best_genome = genome
                patience_left = bundle.scenario.evolution.patience
            else:
                patience_left -= 1

            if patience_left <= 0:
                stop_reason = f"validation score plateaued for {bundle.scenario.evolution.patience} generations"
                break

            failure_patterns = self.nucleus.failure_analyzer.analyze(current_result)
            plan = self.nucleus.plan(
                scenario=bundle.scenario,
                genome=genome,
                generation=generation,
                last_score=current_result.promotion_score,
                best_score=best_result.promotion_score if best_result else -1.0,
                failure_patterns=failure_patterns,
                budget_remaining=budget.remaining_usd,
                lineage_summary=lineage.summary(),
            )
            self._write_json(generation_dir / "mutation_plan.json", plan.model_dump(mode="json"))
            lineage.record_mutation_plan(
                generation, plan.summary, len(plan.proposed_mutations)
            )

            if not plan.proposed_mutations:
                stop_reason = "Nucleus recommended freeze"
                break

            promoted_this_generation = False
            candidate_result = current_result
            for index, mutation in enumerate(plan.proposed_mutations):
                validation = self.guardian.validate_mutation(
                    mutation, genome=genome, scenario=bundle.scenario
                )
                if not validation.allowed:
                    lineage.record_rejected_mutation(
                        generation,
                        mutation.mutation_type,
                        mutation.target,
                        validation.reason,
                    )
                    continue

                mutated_genome = self.guardian.apply_mutation_safely(genome, mutation)
                mutation_dir = generation_dir / f"mutation_{index:02d}"
                mutation_dir.mkdir(parents=True, exist_ok=True)
                save_genome(mutated_genome, mutation_dir / "genome.yaml")
                mutated_result = self._run_and_evaluate(
                    mutated_genome, bundle, mutation_dir, "candidate"
                )
                budget.record(mutated_result.cost_estimate)
                self._write_json(
                    mutation_dir / "eval_result.json",
                    mutated_result.model_dump(mode="json"),
                )

                if self.guardian.should_promote(
                    candidate_result.promotion_score,
                    mutated_result.promotion_score,
                    bundle.scenario.evolution.min_delta,
                ):
                    lineage.record_promoted_mutation(
                        generation,
                        mutation.mutation_type,
                        mutation.target,
                        candidate_result.promotion_score,
                        mutated_result.promotion_score,
                        mutation.rationale,
                    )
                    genome = mutated_genome
                    candidate_result = mutated_result
                    promoted_this_generation = True
                    if best_result is None or self.guardian.should_promote(
                        best_result.promotion_score,
                        mutated_result.promotion_score,
                        bundle.scenario.evolution.min_delta,
                    ):
                        best_result = mutated_result
                        best_genome = mutated_genome
                        patience_left = bundle.scenario.evolution.patience
                else:
                    self.guardian.rollback(f"generation_{generation:03d}_mutation_{index:02d}")
                    lineage.record_rolled_back_mutation(
                        generation,
                        mutation.mutation_type,
                        mutation.target,
                        candidate_result.promotion_score,
                        mutated_result.promotion_score,
                        "fitness did not improve enough for promotion",
                    )

            if not promoted_this_generation:
                patience_left -= 1

        if best_result is None or baseline_result is None:
            raise RuntimeError("Evolution produced no evaluation results")

        frozen_path = run_dir / "frozen_genome.yaml"
        save_genome(best_genome, frozen_path)
        final_dir = run_dir / "final_evaluation"
        final_dir.mkdir(exist_ok=True)
        final_result = self._run_and_evaluate(best_genome, bundle, final_dir, "frozen")
        self._write_json(final_dir / "eval_result.json", final_result.model_dump(mode="json"))
        lineage.record_freeze(
            len(history) - 1,
            final_result.promotion_score,
            stop_reason,
        )
        report_path = ReportBuilder().write_report(
            run_dir=run_dir,
            baseline=baseline_result,
            final=final_result,
            frozen_genome=best_genome,
            lineage=lineage,
            frozen_path=frozen_path,
        )
        return EvolutionRunResult(
            run_dir=run_dir,
            baseline_score=baseline_result.promotion_score,
            final_score=final_result.promotion_score,
            frozen_genome_path=frozen_path,
            report_path=report_path,
        )

    def _run_and_evaluate(
        self,
        genome: Genome,
        bundle: ScenarioBundle,
        generation_dir: Path,
        label: str,
    ) -> EvaluationResult:
        workspace_dir = generation_dir / f"{label}_workspace"
        harness = self.harness_builder.materialize(
            genome,
            bundle.scenario,
            workspace_dir=workspace_dir,
        )
        train_runs = [self.harness_runner.run_case(harness, case) for case in bundle.train_cases]
        validation_runs = [
            self.harness_runner.run_case(harness, case) for case in bundle.validation_cases
        ]
        self._write_runs(generation_dir, label, train_runs + validation_runs)
        return self.guardian.evaluate_candidate(
            genome,
            bundle.scenario,
            train_runs,
            validation_runs,
        )

    def _write_config_snapshot(self, run_dir: Path, bundle: ScenarioBundle) -> None:
        snapshot = {
            "scenario_path": str(bundle.path),
            "scenario": bundle.scenario.model_dump(mode="json"),
            "train_cases": [case.model_dump(mode="json") for case in bundle.train_cases],
            "validation_cases": [
                case.model_dump(mode="json") for case in bundle.validation_cases
            ],
            "settings": self.settings.model_dump(mode="json"),
        }
        (run_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(snapshot, sort_keys=False),
            encoding="utf-8",
        )

    def _write_runs(
        self, generation_dir: Path, label: str, runs: list[HarnessRunResult]
    ) -> None:
        outputs_dir = generation_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        traces_path = generation_dir / f"{label}_traces.jsonl"
        with traces_path.open("w", encoding="utf-8") as traces:
            for run in runs:
                (outputs_dir / f"{label}_{run.case_id}.md").write_text(
                    run.final_output,
                    encoding="utf-8",
                )
                traces.write(json.dumps(run.model_dump(mode="json"), sort_keys=True) + "\n")

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _evaluation_summary(self, result: EvaluationResult) -> str:
        if result.failures:
            return "; ".join(result.failures[:3])
        return "Candidate satisfied visible requirements with acceptable complexity."
