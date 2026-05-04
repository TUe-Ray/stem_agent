from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from stemos.config import Settings, load_settings
from stemos.evolution.control import EvolutionControl, scenario_hash_from_file
from stemos.evolution.lineage import LineageLog
from stemos.evolution.reports import ReportBuilder
from stemos.genome.loader import load_default_genome
from stemos.genome.loader import load_genome
from stemos.genome.models import Genome
from stemos.genome.serializer import save_genome
from stemos.harness.builder import HarnessBuilder
from stemos.harness.runner import HarnessRunResult, HarnessRunner
from stemos.kernel.budget import BudgetTracker
from stemos.kernel.evaluator import EvaluationResult
from stemos.kernel.guardian import Guardian
from stemos.kernel.versioning import GitProvenance, GitRunConfig
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
    status: str = "FROZEN"


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
        model_client = ModelClient(
            model=self.settings.model,
            offline=self.settings.offline_mode,
            endpoint=self.settings.openai_endpoint,
        )
        self.model_client = model_client
        self.nucleus = NucleusCommander(
            scenario_interpreter=ScenarioInterpreter(model_client),
            failure_analyzer=FailureAnalyzer(),
            mutation_planner=MutationPlanner(model_client),
        )
        self.guardian = guardian or Guardian()
        self.harness_builder = harness_builder or HarnessBuilder()
        self.harness_runner = harness_runner or HarnessRunner()
        self.runs_root = Path(runs_root)

    def resume(
        self,
        run_id: str,
        *,
        git_branch: bool = False,
        git_commit: bool = False,
        git_push: bool = False,
    ) -> EvolutionRunResult:
        run_dir = self.runs_root / run_id
        scenario_path = self._scenario_path_from_run(run_dir)
        return self.evolve(
            scenario_path,
            run_id,
            git_branch=git_branch,
            git_commit=git_commit,
            git_push=git_push,
            resume=True,
        )

    def freeze_now(
        self,
        run_id: str,
        *,
        git_branch: bool = False,
        git_commit: bool = False,
    ) -> EvolutionRunResult:
        run_dir = self.runs_root / run_id
        control = EvolutionControl(run_dir, run_id)
        state = control.load_checkpoint()
        scenario_path = self._scenario_path_from_run(run_dir)
        bundle = load_scenario(scenario_path)
        if state.scenario_hash != scenario_hash_from_file(bundle.path):
            raise RuntimeError("Checkpoint scenario_hash does not match current scenario")
        baseline_genome = load_genome(run_dir / "baseline_genome.yaml")
        best_genome = load_genome(state.best_genome_path)
        if not state.baseline_eval or not state.best_eval:
            raise RuntimeError("Cannot freeze-now before a verified baseline and best result")
        lineage = LineageLog(run_dir)
        lineage.record("freeze_now", generation=state.generation, summary="Freeze-now requested")
        result = self._freeze_best(
            bundle=bundle,
            run_dir=run_dir,
            lineage=lineage,
            generation=state.generation,
            baseline_result=EvaluationResult.model_validate(state.baseline_eval),
            best_result=EvaluationResult.model_validate(state.best_eval),
            baseline_genome=baseline_genome,
            best_genome=best_genome,
            stop_reason="freeze-now requested",
            status="FROZEN",
        )
        control.set_status("FROZEN", command="freeze_now", message="freeze-now requested")
        git = GitProvenance(
            GitRunConfig(
                run_id=run_id,
                run_dir=run_dir,
                branch_enabled=git_branch,
                commit_enabled=git_commit,
            )
        )
        git.start()
        git.commit_safe("StemOS freeze-now", [run_dir])
        return result

    def evolve(
        self,
        scenario_path: str | Path,
        run_id: str,
        *,
        git_branch: bool = False,
        git_commit: bool = False,
        git_push: bool = False,
        resume: bool = False,
    ) -> EvolutionRunResult:
        bundle = load_scenario(scenario_path)
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        control = EvolutionControl(run_dir, run_id)
        git = GitProvenance(
            GitRunConfig(
                run_id=run_id,
                run_dir=run_dir,
                branch_enabled=git_branch,
                commit_enabled=git_commit,
                push_enabled=git_push,
            )
        )
        git_start = git.start()
        if not git_start.allowed:
            raise RuntimeError(git_start.reason)

        scenario_hash = scenario_hash_from_file(bundle.path)
        lineage = LineageLog(run_dir, reset=not resume)
        budget = BudgetTracker(max_cost_usd=bundle.scenario.evolution.max_cost_usd)
        existing_control = control.read()
        if existing_control.command not in {"pause", "freeze_now", "abort"}:
            control.set_status("RUNNING")

        if resume:
            state = control.load_checkpoint()
            if state.scenario_hash != scenario_hash:
                raise RuntimeError("Checkpoint scenario_hash does not match current scenario")
            genome = load_genome(state.current_genome_path)
            best_genome = load_genome(state.best_genome_path)
            baseline_genome = load_genome(run_dir / "baseline_genome.yaml")
            baseline_result = (
                EvaluationResult.model_validate(state.baseline_eval)
                if state.baseline_eval
                else None
            )
            best_result = (
                EvaluationResult.model_validate(state.best_eval)
                if state.best_eval
                else None
            )
            patience_left = state.patience_left
            start_generation = state.next_generation
            stop_reason = state.stop_reason or "maximum generations reached"
            lineage.record("resume", generation=start_generation, summary="Resumed from checkpoint")
            git.commit_safe("StemOS resume event", [run_dir])
        else:
            self._write_config_snapshot(run_dir, bundle)
            diagnosis = self.nucleus.diagnose(bundle.scenario)
            genome = self.nucleus.attach_diagnosis(load_default_genome(), diagnosis, bundle.scenario)
            baseline_genome = genome
            save_genome(genome, run_dir / "baseline_genome.yaml")
            best_genome = genome
            best_result = None
            baseline_result = None
            patience_left = bundle.scenario.evolution.patience
            start_generation = 0
            stop_reason = "maximum generations reached"

        history: list[EvaluationResult] = []

        for generation in range(start_generation, bundle.scenario.evolution.max_generations):
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

            self._save_checkpoint(
                control,
                scenario_hash,
                generation,
                generation,
                "baseline_evaluation_complete" if baseline_result is current_result else "evaluation_complete",
                genome,
                best_genome,
                baseline_result,
                best_result,
                patience_left,
                stop_reason,
            )
            git.commit_safe(
                "StemOS baseline evaluation complete"
                if generation == 0 and baseline_result is current_result
                else f"StemOS generation {generation} evaluation",
                [run_dir],
            )
            control_result = self._handle_control_safe_point(
                control=control,
                git=git,
                bundle=bundle,
                run_dir=run_dir,
                generation=generation,
                genome=genome,
                best_genome=best_genome,
                baseline_genome=baseline_genome,
                baseline_result=baseline_result,
                best_result=best_result,
                patience_left=patience_left,
                scenario_hash=scenario_hash,
                stop_reason=stop_reason,
                lineage=lineage,
            )
            if control_result:
                return control_result

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
                lineage.record_proposed_mutation(
                    generation,
                    mutation.mutation_type,
                    mutation.target,
                    mutation.rationale,
                    mutation.expected_improvement,
                    mutation.risk,
                )
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
                    self._save_checkpoint(
                        control,
                        scenario_hash,
                        generation,
                        generation,
                        "mutation_rejected",
                        genome,
                        best_genome,
                        baseline_result,
                        best_result,
                        patience_left,
                        stop_reason,
                    )
                    git.commit_safe(
                        f"StemOS rejected {mutation.mutation_type}", [run_dir]
                    )
                    control_result = self._handle_control_safe_point(
                        control=control,
                        git=git,
                        bundle=bundle,
                        run_dir=run_dir,
                        generation=generation,
                        genome=genome,
                        best_genome=best_genome,
                        baseline_genome=baseline_genome,
                        baseline_result=baseline_result,
                        best_result=best_result,
                        patience_left=patience_left,
                        scenario_hash=scenario_hash,
                        stop_reason=stop_reason,
                        lineage=lineage,
                    )
                    if control_result:
                        return control_result
                    continue

                mutation_dir = generation_dir / f"mutation_{index:02d}"
                mutation_dir.mkdir(parents=True, exist_ok=True)
                if mutation.mutation_type in {"create_tool", "edit_tool"}:
                    activation, activated_mutation = self.guardian.activate_generated_tool(
                        mutation, mutation_dir / "generated_tools"
                    )
                    if not activation.allowed:
                        lineage.record_rejected_mutation(
                            generation,
                            mutation.mutation_type,
                            mutation.target,
                            activation.reason,
                        )
                        self._save_checkpoint(
                            control,
                            scenario_hash,
                            generation,
                            generation,
                            "mutation_rejected",
                            genome,
                            best_genome,
                            baseline_result,
                            best_result,
                            patience_left,
                            stop_reason,
                        )
                        git.commit_safe(
                            f"StemOS rejected {mutation.mutation_type}", [run_dir]
                        )
                        control_result = self._handle_control_safe_point(
                            control=control,
                            git=git,
                            bundle=bundle,
                            run_dir=run_dir,
                            generation=generation,
                            genome=genome,
                            best_genome=best_genome,
                            baseline_genome=baseline_genome,
                            baseline_result=baseline_result,
                            best_result=best_result,
                            patience_left=patience_left,
                            scenario_hash=scenario_hash,
                            stop_reason=stop_reason,
                            lineage=lineage,
                        )
                        if control_result:
                            return control_result
                        continue
                    mutation = activated_mutation

                mutated_genome = self.guardian.apply_mutation_safely(genome, mutation)
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
                    self._save_checkpoint(
                        control,
                        scenario_hash,
                        generation,
                        generation,
                        "mutation_promoted",
                        genome,
                        best_genome,
                        baseline_result,
                        best_result,
                        patience_left,
                        stop_reason,
                    )
                    git.commit_safe(
                        f"StemOS promoted {mutation.mutation_type}", [run_dir]
                    )
                    control_result = self._handle_control_safe_point(
                        control=control,
                        git=git,
                        bundle=bundle,
                        run_dir=run_dir,
                        generation=generation,
                        genome=genome,
                        best_genome=best_genome,
                        baseline_genome=baseline_genome,
                        baseline_result=baseline_result,
                        best_result=best_result,
                        patience_left=patience_left,
                        scenario_hash=scenario_hash,
                        stop_reason=stop_reason,
                        lineage=lineage,
                    )
                    if control_result:
                        return control_result
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
                    self._save_checkpoint(
                        control,
                        scenario_hash,
                        generation,
                        generation,
                        "mutation_rolled_back",
                        genome,
                        best_genome,
                        baseline_result,
                        best_result,
                        patience_left,
                        stop_reason,
                    )
                    git.commit_safe(
                        f"StemOS rolled back {mutation.mutation_type}", [run_dir]
                    )
                    control_result = self._handle_control_safe_point(
                        control=control,
                        git=git,
                        bundle=bundle,
                        run_dir=run_dir,
                        generation=generation,
                        genome=genome,
                        best_genome=best_genome,
                        baseline_genome=baseline_genome,
                        baseline_result=baseline_result,
                        best_result=best_result,
                        patience_left=patience_left,
                        scenario_hash=scenario_hash,
                        stop_reason=stop_reason,
                        lineage=lineage,
                    )
                    if control_result:
                        return control_result

            if not promoted_this_generation:
                patience_left -= 1
            self._save_checkpoint(
                control,
                scenario_hash,
                generation,
                generation + 1,
                "generation_complete",
                genome,
                best_genome,
                baseline_result,
                best_result,
                patience_left,
                stop_reason,
            )
            git.commit_safe(f"StemOS generation {generation} complete", [run_dir])
            control_result = self._handle_control_safe_point(
                control=control,
                git=git,
                bundle=bundle,
                run_dir=run_dir,
                generation=generation,
                genome=genome,
                best_genome=best_genome,
                baseline_genome=baseline_genome,
                baseline_result=baseline_result,
                best_result=best_result,
                patience_left=patience_left,
                scenario_hash=scenario_hash,
                stop_reason=stop_reason,
                lineage=lineage,
            )
            if control_result:
                return control_result

        if best_result is None or baseline_result is None:
            raise RuntimeError("Evolution produced no evaluation results")

        result = self._freeze_best(
            bundle=bundle,
            run_dir=run_dir,
            lineage=lineage,
            generation=len(history) - 1,
            baseline_result=baseline_result,
            best_result=best_result,
            baseline_genome=baseline_genome,
            best_genome=best_genome,
            stop_reason=stop_reason,
            status="FROZEN",
        )
        control.set_status("FROZEN", message=stop_reason)
        git.commit_safe("StemOS final freeze", [run_dir])
        if git_push:
            push_result = git.push()
            if not push_result.allowed:
                raise RuntimeError(push_result.reason)
        return result

    def _freeze_best(
        self,
        *,
        bundle: ScenarioBundle,
        run_dir: Path,
        lineage: LineageLog,
        generation: int,
        baseline_result: EvaluationResult,
        best_result: EvaluationResult,
        baseline_genome: Genome,
        best_genome: Genome,
        stop_reason: str,
        status: str,
    ) -> EvolutionRunResult:
        frozen_path = run_dir / "frozen_genome.yaml"
        save_genome(best_genome, frozen_path)
        final_dir = run_dir / "final_evaluation"
        final_dir.mkdir(exist_ok=True)
        final_result = self._run_and_evaluate(best_genome, bundle, final_dir, "frozen")
        self._write_json(final_dir / "eval_result.json", final_result.model_dump(mode="json"))
        self._write_json(run_dir / "run_metadata.json", self._run_metadata(final_result))
        lineage.record_freeze(
            generation,
            final_result.promotion_score,
            stop_reason,
        )
        report_path = ReportBuilder().write_report(
            run_dir=run_dir,
            baseline=baseline_result,
            final=final_result,
            baseline_genome=baseline_genome,
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
            status=status,
        )

    def _save_checkpoint(
        self,
        control: EvolutionControl,
        scenario_hash: str,
        generation: int,
        next_generation: int,
        phase: str,
        genome: Genome,
        best_genome: Genome,
        baseline_result: EvaluationResult | None,
        best_result: EvaluationResult | None,
        patience_left: int,
        stop_reason: str,
    ) -> None:
        control.save_checkpoint(
            scenario_hash=scenario_hash,
            generation=generation,
            next_generation=next_generation,
            phase=phase,
            current_genome=genome,
            best_genome=best_genome,
            baseline_eval=baseline_result,
            best_eval=best_result,
            patience_left=patience_left,
            stop_reason=stop_reason,
        )

    def _handle_control_safe_point(
        self,
        *,
        control: EvolutionControl,
        git: GitProvenance,
        bundle: ScenarioBundle,
        run_dir: Path,
        generation: int,
        genome: Genome,
        best_genome: Genome,
        baseline_genome: Genome,
        baseline_result: EvaluationResult | None,
        best_result: EvaluationResult | None,
        patience_left: int,
        scenario_hash: str,
        stop_reason: str,
        lineage: LineageLog,
    ) -> EvolutionRunResult | None:
        state = control.read()
        if state.command == "resume":
            control.clear_command("RUNNING")
            lineage.record("resume", generation=generation, summary="Resume control event observed")
            git.commit_safe("StemOS resume event", [run_dir])
            return None
        if state.command == "pause":
            self._save_checkpoint(
                control,
                scenario_hash,
                generation,
                generation,
                "safe_pause",
                genome,
                best_genome,
                baseline_result,
                best_result,
                patience_left,
                stop_reason,
            )
            lineage.record("pause", generation=generation, summary="Paused at safe checkpoint")
            control.set_status("PAUSED", command="pause", message="paused at safe checkpoint")
            git.commit_safe("StemOS safe pause checkpoint", [run_dir])
            return self._non_frozen_result(run_dir, baseline_result, best_result, "PAUSED")
        if state.command == "freeze_now":
            if baseline_result is None or best_result is None:
                return None
            lineage.record("freeze_now", generation=generation, summary="Freeze-now requested")
            result = self._freeze_best(
                bundle=bundle,
                run_dir=run_dir,
                lineage=lineage,
                generation=generation,
                baseline_result=baseline_result,
                best_result=best_result,
                baseline_genome=baseline_genome,
                best_genome=best_genome,
                stop_reason="freeze-now requested",
                status="FROZEN",
            )
            control.set_status("FROZEN", command="freeze_now", message="freeze-now requested")
            git.commit_safe("StemOS freeze-now", [run_dir])
            return result
        if state.command == "abort":
            self._save_checkpoint(
                control,
                scenario_hash,
                generation,
                generation,
                "abort",
                genome,
                best_genome,
                baseline_result,
                best_result,
                patience_left,
                stop_reason,
            )
            lineage.record("abort", generation=generation, summary="Abort requested; candidate not promoted")
            control.set_status("ABORTED", command="abort", message="abort requested")
            git.commit_safe("StemOS abort checkpoint", [run_dir])
            return self._non_frozen_result(run_dir, baseline_result, best_result, "ABORTED")
        return None

    def _non_frozen_result(
        self,
        run_dir: Path,
        baseline_result: EvaluationResult | None,
        best_result: EvaluationResult | None,
        status: str,
    ) -> EvolutionRunResult:
        return EvolutionRunResult(
            run_dir=run_dir,
            baseline_score=baseline_result.promotion_score if baseline_result else 0.0,
            final_score=best_result.promotion_score if best_result else 0.0,
            frozen_genome_path=run_dir / "frozen_genome.yaml",
            report_path=run_dir / "report.md",
            status=status,
        )

    def _scenario_path_from_run(self, run_dir: Path) -> Path:
        config_path = run_dir / "config_snapshot.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing run config snapshot: {config_path}")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not data.get("scenario_path"):
            raise RuntimeError(f"Run config does not contain scenario_path: {config_path}")
        return Path(data["scenario_path"])

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

    def _run_metadata(self, final_result: EvaluationResult) -> dict:
        mode = "offline deterministic" if self.settings.offline_mode else "openai-backed"
        return {
            "run_mode": mode,
            "model": self.settings.model,
            "endpoint": self.model_client.endpoint,
            "offline_mode": self.settings.offline_mode,
            "fallback_used": self.model_client.fallback_used,
            "model_calls": self.model_client.model_calls,
            "structured_output_repairs": self.model_client.structured_output_repairs,
            "total_estimated_cost": final_result.cost_estimate,
        }
