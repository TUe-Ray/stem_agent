from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel

from stem_agent.config import Settings, load_settings
from stem_agent.evolution.control import EvolutionControl, scenario_hash_from_file
from stem_agent.evolution.lineage import LineageLog
from stem_agent.evolution.reports import ReportBuilder
from stem_agent.genome.loader import load_default_genome
from stem_agent.genome.loader import load_genome
from stem_agent.genome.models import Genome
from stem_agent.genome.serializer import save_genome
from stem_agent.harness.builder import HarnessBuilder
from stem_agent.harness.runner import HarnessRunResult, HarnessRunner
from stem_agent.harness.role_runner import RoleRunner
from stem_agent.kernel.budget import BudgetTracker
from stem_agent.kernel.convergence import ConvergenceEngine
from stem_agent.kernel.evaluator import EvaluationResult, GuardianFitnessEvaluator
from stem_agent.kernel.guardian import Guardian
from stem_agent.kernel.signal_policy import NucleusSignal
from stem_agent.kernel.versioning import GenomeArchive, GitProvenance, GitRunConfig
from stem_agent.nucleus.commander import NucleusCommander
from stem_agent.nucleus.failure_analyzer import FailureAnalyzer
from stem_agent.nucleus.model_client import ModelClient
from stem_agent.nucleus.mutation_planner import MutationPlanner
from stem_agent.nucleus.nucleus import nucleus_signal_to_dict, select_operator
from stem_agent.nucleus.operators import ZeroOrderMutation
from stem_agent.nucleus.schemas import MutationProposal
from stem_agent.nucleus.scenario_interpreter import ScenarioInterpreter
from stem_agent.scenarios.loader import ScenarioBundle, load_scenario
from stem_agent.skills.awm_bridge import AWMBridge
from stem_agent.skills.extractor import SkillExtractor
from stem_agent.skills.library import SkillLibrary


class EvolutionRunResult(BaseModel):
    run_dir: Path
    baseline_score: float
    final_score: float
    frozen_genome_path: Path
    report_path: Path
    status: str = "FROZEN"


class EvolutionLoop:
    def _combine_external_signal(
        self,
        *,
        internal_result: EvaluationResult,
        external_result: EvaluationResult | None,
        signal_mode: str,
        external_signal_weight: float,
    ) -> EvaluationResult:
        if signal_mode != "internal_plus_external_score" or external_result is None:
            return internal_result.model_copy(update={"metrics": {**internal_result.metrics, "internal_promotion_score": float(internal_result.promotion_score), "external_dev_score": 0.0, "external_signal_weight": 0.0, "signal_mode": signal_mode}, "split_policy": f"{internal_result.split_policy}+{signal_mode}"})
        external_score = float(external_result.train_score)
        combined = (1.0 - external_signal_weight) * float(internal_result.promotion_score) + external_signal_weight * external_score
        return internal_result.model_copy(update={"score": combined, "promotion_score": combined, "metrics": {**internal_result.metrics, "internal_promotion_score": float(internal_result.promotion_score), "external_dev_score": external_score, "external_signal_weight": float(external_signal_weight), "signal_mode": signal_mode}, "split_policy": "internal_plus_external_score_weighted"})

    def _fitness_vector(self, result: EvaluationResult) -> dict[str, float]:
        return {
            "promotion_score": float(result.promotion_score),
            "task_quality": float(result.task_quality),
            "train_score": float(result.train_score),
            "validation_score": float(result.validation_score or 0.0),
            "cost_efficiency": float(result.cost_efficiency),
            "safety_score": float(result.safety_score),
            "complexity_penalty": float(result.complexity_penalty),
            "stability_score": float(result.stability_score),
            "cost_estimate": float(result.cost_estimate),
        }

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        guardian: Guardian | None = None,
        harness_builder: HarnessBuilder | None = None,
        harness_runner: HarnessRunner | None = None,
        runs_root: str | Path = "runs",
        skill_library_path: str | Path | None = None,
    ):
        self.settings = settings or load_settings()
        model_client = ModelClient(
            model=self.settings.model,
            test_mode=self.settings.test_mode,
            endpoint=self.settings.openai_endpoint,
        )
        self.model_client = model_client
        default_harness_runner = HarnessRunner(RoleRunner(model_client))
        self.nucleus = NucleusCommander(
            scenario_interpreter=ScenarioInterpreter(model_client),
            failure_analyzer=FailureAnalyzer(),
            mutation_planner=MutationPlanner(model_client),
        )
        self.harness_runner = harness_runner or default_harness_runner
        self.guardian = guardian or Guardian(
            evaluator=GuardianFitnessEvaluator(model_client),
            harness_runner=self.harness_runner,
        )
        self.harness_builder = harness_builder or HarnessBuilder()
        self.runs_root = Path(runs_root)
        default_skill_path = self.runs_root.parent / "skills" / "atoms.jsonl"
        self.skill_library = SkillLibrary(skill_library_path or default_skill_path)
        self.skill_extractor = SkillExtractor()
        self.awm_bridge = AWMBridge()

    def resume(
        self,
        run_id: str,
        *,
        git_branch: bool = False,
        git_commit: bool = False,
        git_push: bool = False,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
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
            event_sink=event_sink,
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
        git.commit_safe("stem_agent freeze-now", [run_dir])
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
        signal_policy_override: dict[str, bool] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvolutionRunResult:
        bundle = load_scenario(scenario_path)
        if signal_policy_override:
            bundle.scenario.signal_policy = bundle.scenario.signal_policy.__class__(
                **{
                    **bundle.scenario.signal_policy.__dict__,
                    **signal_policy_override,
                }
            )
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.model_client.configure_run(run_dir)
        self.guardian.configure_run(run_dir)
        self.guardian.configure_signal_policy(bundle.scenario.signal_policy)
        self.guardian.configure_hidden_evaluation(bundle.path)
        self.guardian.configure_validation_cases(bundle.validation_cases)
        self._emit_event(
            event_sink,
            {
                "event": "evolution_start",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "scenario": bundle.scenario.name,
                "scenario_path": str(bundle.path),
                "task_class": bundle.scenario.scenario.task_class,
                "resume": resume,
                "model": self.settings.model,
                "endpoint": self.settings.openai_endpoint,
                "train_case_count": len(bundle.train_cases),
                "validation_case_count": len(bundle.validation_cases),
                "hidden_case_count": self._hidden_case_count(),
                "max_generations": bundle.scenario.convergence_policy.max_generations,
                "max_mutations_per_generation": bundle.scenario.evolution.max_mutations_per_generation,
                "patience": bundle.scenario.evolution.patience,
                "min_delta": bundle.scenario.evolution.min_delta,
                "max_cost_usd": bundle.scenario.evolution.max_cost_usd,
            },
        )
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
            archive = GenomeArchive.from_snapshot(state.archive, run_dir=run_dir)
            current_parent_id = state.current_parent_id
            stagnation_count = state.stagnation_count
            start_generation = state.next_generation
            stop_reason = state.stop_reason or "maximum generations reached"
            lineage.record("resume", generation=start_generation, summary="Resumed from checkpoint")
            git.commit_safe("stem_agent resume event", [run_dir])
        else:
            self._write_config_snapshot(run_dir, bundle)
            diagnosis = self.nucleus.diagnose(bundle.scenario)
            self._emit_event(
                event_sink,
                {
                    "event": "diagnosis_complete",
                    "initial_evaluation_hypothesis": diagnosis.initial_evaluation_hypothesis,
                    "requirement_count": len(bundle.scenario.expected_output.requirements),
                    "constraint_count": len(bundle.scenario.constraints),
                },
            )
            seed_genome = (
                self._gsm8k_baseline_genome(bundle.scenario.name)
                if bundle.scenario.name == "gsm8k_mini"
                else load_default_genome()
            )
            genome = self.nucleus.attach_diagnosis(seed_genome, diagnosis, bundle.scenario)
            baseline_genome = genome
            save_genome(genome, run_dir / "baseline_genome.yaml")
            best_genome = genome
            best_result = None
            baseline_result = None
            patience_left = bundle.scenario.evolution.patience
            archive = GenomeArchive(max_size=10, run_dir=run_dir)
            current_parent_id = None
            stagnation_count = 0
            start_generation = 0
            stop_reason = "maximum generations reached"
        self._active_archive = archive
        self._active_parent_id = current_parent_id
        self._active_stagnation_count = stagnation_count
        last_archive_best_score = archive.best()[2] if len(archive) else None

        history: list[EvaluationResult] = []
        signal_history: list[NucleusSignal] = self._load_signal_history(lineage)
        convergence = ConvergenceEngine(
            bundle.scenario.convergence_policy,
            run_dir=run_dir,
            hidden_baseline_score=self.guardian.hidden_baseline_score,
        )
        force_zero_order_next = False
        pending_awm_candidates: list[MutationProposal] = []

        for generation in range(start_generation, bundle.scenario.convergence_policy.max_generations):
            generation_dir = run_dir / f"generation_{generation:03d}"
            generation_dir.mkdir(parents=True, exist_ok=True)
            save_genome(genome, generation_dir / "genome.yaml")
            tokens_before_generation = self._total_tokens_used(run_dir)
            self._emit_event(
                event_sink,
                {
                    "event": "generation_start",
                    "generation": generation,
                    "genome_version": genome.genome_version,
                },
            )

            current_result = self._run_and_evaluate(
                genome,
                bundle,
                generation_dir,
                "current",
                generation=generation,
                event_sink=event_sink,
            )
            history.append(current_result)
            budget.record(current_result.cost_estimate)
            self._write_json(generation_dir / "eval_result.json", current_result.model_dump(mode="json"))
            self._emit_event(
                event_sink,
                {
                    "event": "evaluation_complete",
                    "generation": generation,
                    "label": "current",
                    "score": current_result.promotion_score,
                    "train_score": current_result.train_score,
                    "validation_score": current_result.validation_score,
                },
            )

            if baseline_result is None:
                baseline_result = current_result
                self._write_json(
                    run_dir / "baseline_genome_score.json",
                    {
                        "genome_id": current_parent_id,
                        "score": baseline_result.promotion_score,
                        "train_score": baseline_result.train_score,
                        "validation_score": baseline_result.validation_score,
                    },
                )

            if current_parent_id is None:
                current_fitness_vector = self._fitness_vector(current_result)
                current_parent_id = archive.add(
                    genome.model_dump(mode="json"),
                    current_result.promotion_score,
                    generation,
                    parent_id=None,
                    fitness_vector=current_fitness_vector,
                )
                self._active_parent_id = current_parent_id
                if baseline_result is current_result:
                    self._write_json(
                        run_dir / "baseline_genome_score.json",
                        {
                            "genome_id": current_parent_id,
                            "score": baseline_result.promotion_score,
                            "train_score": baseline_result.train_score,
                            "validation_score": baseline_result.validation_score,
                        },
                    )
                    self.guardian.hidden_baseline_score = self._run_hidden_evaluation(
                        genome,
                        bundle,
                        generation_dir / "hidden_baseline_workspace",
                        current_parent_id,
                        generation=generation,
                        label="baseline",
                        event_sink=event_sink,
                    )
                    convergence.hidden_baseline_score = self.guardian.hidden_baseline_score

            summary = self._evaluation_summary(current_result)
            previous_eval_score = history[-2].promotion_score if len(history) >= 2 else 0.0
            evaluation_signal = bundle.scenario.signal_policy.build_nucleus_signal(
                generation=generation,
                mutation_type="evaluation",
                current_score=current_result.promotion_score,
                previous_score=previous_eval_score,
            )
            signal_history.append(evaluation_signal)
            lineage.record_evaluation(
                generation,
                current_result.promotion_score,
                summary,
                nucleus_signal=nucleus_signal_to_dict(evaluation_signal),
                fitness_vector=self._fitness_vector(current_result),
            )

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
                "stem_agent baseline evaluation complete"
                if generation == 0 and baseline_result is current_result
                else f"stem_agent generation {generation} evaluation",
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

            failure_pattern_objects = self.nucleus.failure_analyzer.analyze_patterns(current_result)
            failure_patterns = [pattern.kind for pattern in failure_pattern_objects]
            operator = ZeroOrderMutation() if force_zero_order_next else select_operator(archive, stagnation_count)
            force_zero_order_next = False
            reusable_skills = self._retrieve_reusable_skills(
                bundle=bundle,
                run_id=run_id,
            )
            injected_skill_ids = [skill.id for skill in reusable_skills]
            plan = self.nucleus.plan(
                scenario=bundle.scenario,
                genome=genome,
                generation=generation,
                last_score=current_result.promotion_score,
                best_score=best_result.promotion_score if best_result else -1.0,
                failure_patterns=failure_patterns,
                structured_failure_patterns=failure_pattern_objects,
                budget_remaining=budget.remaining_usd,
                lineage_summary=lineage.summary(),
                operator=operator,
                archive=archive,
                mutation_history=signal_history,
                signal_policy=bundle.scenario.signal_policy,
                reusable_skills=reusable_skills,
            )
            if pending_awm_candidates:
                plan = plan.model_copy(
                    update={
                        "summary": "AWM candidates queued before Nucleus plan. " + plan.summary,
                        "proposed_mutations": pending_awm_candidates + plan.proposed_mutations,
                    }
                )
                pending_awm_candidates = []
            self._write_json(generation_dir / "mutation_plan.json", plan.model_dump(mode="json"))
            lineage.record_mutation_plan(
                generation,
                plan.summary,
                len(plan.proposed_mutations),
                nucleus_signal=nucleus_signal_to_dict(signal_history[-1]) if signal_history else None,
                fitness_vector=self._fitness_vector(current_result),
            )
            self._emit_event(
                event_sink,
                {
                    "event": "mutation_plan",
                    "generation": generation,
                    "summary": plan.summary,
                    "proposed_count": len(plan.proposed_mutations),
                    "failure_patterns": plan.failure_patterns,
                    "operator_type": operator.name,
                },
            )

            if not plan.proposed_mutations:
                stop_reason = "Nucleus recommended freeze"
                self._emit_event(
                    event_sink,
                    {
                        "event": "freeze_recommended",
                        "generation": generation,
                        "reason": stop_reason,
                    },
                )
                break

            promoted_this_generation = False
            candidate_result = current_result
            generation_terminal_result = current_result
            for index, mutation in enumerate(plan.proposed_mutations):
                lineage.record_proposed_mutation(
                    generation,
                    mutation.mutation_type,
                    mutation.target,
                    mutation.rationale,
                    mutation.expected_improvement,
                    mutation.risk,
                    operator_type=operator.name,
                    origin=mutation.origin,
                )
                self._emit_event(
                    event_sink,
                    {
                        "event": "mutation_proposed",
                        "generation": generation,
                        "index": index,
                        "mutation_type": mutation.mutation_type,
                        "target": mutation.target,
                        "rationale": mutation.rationale,
                        "expected_improvement": mutation.expected_improvement,
                        "risk": mutation.risk,
                        "operator_type": operator.name,
                    },
                )
                validation = self.guardian.validate_mutation(
                    mutation, genome=genome, scenario=bundle.scenario
                )
                if not validation.allowed:
                    rejection_signal = bundle.scenario.signal_policy.build_nucleus_signal(
                        generation=generation,
                        mutation_type=mutation.mutation_type,
                        current_score=candidate_result.promotion_score,
                        previous_score=candidate_result.promotion_score,
                    )
                    signal_history.append(rejection_signal)
                    lineage.record_rejected_mutation(
                        generation,
                        mutation.mutation_type,
                        mutation.target,
                        validation.reason,
                        operator_type=operator.name,
                        nucleus_signal=nucleus_signal_to_dict(rejection_signal),
                    )
                    self._emit_event(
                        event_sink,
                        {
                            "event": "mutation_rejected",
                            "generation": generation,
                            "index": index,
                            "mutation_type": mutation.mutation_type,
                            "target": mutation.target,
                            "reason": validation.reason,
                            "operator_type": operator.name,
                        },
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
                        f"stem_agent rejected {mutation.mutation_type}", [run_dir]
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
                        rejection_signal = bundle.scenario.signal_policy.build_nucleus_signal(
                            generation=generation,
                            mutation_type=mutation.mutation_type,
                            current_score=candidate_result.promotion_score,
                            previous_score=candidate_result.promotion_score,
                        )
                        signal_history.append(rejection_signal)
                        lineage.record_rejected_mutation(
                            generation,
                            mutation.mutation_type,
                            mutation.target,
                            activation.reason,
                            operator_type=operator.name,
                            nucleus_signal=nucleus_signal_to_dict(rejection_signal),
                        )
                        self._emit_event(
                            event_sink,
                            {
                                "event": "mutation_rejected",
                                "generation": generation,
                                "index": index,
                                "mutation_type": mutation.mutation_type,
                                "target": mutation.target,
                                "reason": activation.reason,
                                "operator_type": operator.name,
                            },
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
                            f"stem_agent rejected {mutation.mutation_type}", [run_dir]
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
                safety_validation = self.guardian.validate_candidate_safety(
                    genome,
                    mutated_genome,
                    bundle.scenario,
                    train_cases=[case.model_dump(mode="json") for case in bundle.train_cases],
                    run_dir=run_dir,
                )
                if not safety_validation.allowed:
                    self.guardian.rollback(f"generation_{generation:03d}_mutation_{index:02d}")
                    rejection_signal = bundle.scenario.signal_policy.build_nucleus_signal(
                        generation=generation,
                        mutation_type=mutation.mutation_type,
                        current_score=candidate_result.promotion_score,
                        previous_score=candidate_result.promotion_score,
                    )
                    signal_history.append(rejection_signal)
                    lineage.record_rejected_mutation(
                        generation,
                        mutation.mutation_type,
                        mutation.target,
                        safety_validation.reason,
                        operator_type=operator.name,
                        mutation_rejected_by="safety_validator",
                        nucleus_signal=nucleus_signal_to_dict(rejection_signal),
                    )
                    self._emit_event(
                        event_sink,
                        {
                            "event": "mutation_rejected",
                            "generation": generation,
                            "index": index,
                            "mutation_type": mutation.mutation_type,
                            "target": mutation.target,
                            "reason": safety_validation.reason,
                            "operator_type": operator.name,
                            "mutation_rejected_by": "safety_validator",
                        },
                    )
                    self._save_checkpoint(
                        control,
                        scenario_hash,
                        generation,
                        generation,
                        "mutation_rejected_by_safety_validator",
                        genome,
                        best_genome,
                        baseline_result,
                        best_result,
                        patience_left,
                        stop_reason,
                    )
                    git.commit_safe(
                        f"stem_agent safety-rejected {mutation.mutation_type}", [run_dir]
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
                mutated_result = self._run_and_evaluate(
                    mutated_genome,
                    bundle,
                    mutation_dir,
                    "candidate",
                    generation=generation,
                    mutation_index=index,
                    event_sink=event_sink,
                )
                budget.record(mutated_result.cost_estimate)
                self._write_json(
                    mutation_dir / "eval_result.json",
                    mutated_result.model_dump(mode="json"),
                )
                candidate_genome_id = archive.add(
                    mutated_genome.model_dump(mode="json"),
                    mutated_result.promotion_score,
                    generation,
                    parent_id=current_parent_id,
                    fitness_vector=self._fitness_vector(mutated_result),
                )

                if self.guardian.should_promote(
                    candidate_result.promotion_score,
                    mutated_result.promotion_score,
                    bundle.scenario.evolution.min_delta,
                ):
                    mutation_signal = bundle.scenario.signal_policy.build_nucleus_signal(
                        generation=generation,
                        mutation_type=mutation.mutation_type,
                        current_score=mutated_result.promotion_score,
                        previous_score=candidate_result.promotion_score,
                    )
                    signal_history.append(mutation_signal)
                    hidden_score = self._run_hidden_evaluation(
                        mutated_genome,
                        bundle,
                        mutation_dir / "hidden_workspace",
                        candidate_genome_id,
                        generation=generation,
                        label="candidate",
                        mutation_index=index,
                        event_sink=event_sink,
                    )
                    baseline_hidden = self.guardian.hidden_baseline_score
                    hidden_regression = (
                        baseline_hidden is not None and baseline_hidden - hidden_score > 0.1
                    )
                    extracted_skill_ids = self._extract_and_store_skills(
                        old_genome=genome,
                        new_genome=mutated_genome,
                        mutation_type=mutation.mutation_type,
                        score_before=candidate_result.promotion_score,
                        score_after=mutated_result.promotion_score,
                        run_id=run_id,
                        scenario_name=bundle.scenario.name,
                        generation=generation,
                        domain_tags=bundle.scenario.effective_domain_tags,
                    )
                    lineage.record_promoted_mutation(
                        generation,
                        mutation.mutation_type,
                        mutation.target,
                        candidate_result.promotion_score,
                        mutated_result.promotion_score,
                        mutation.rationale,
                        operator_type=operator.name,
                        hidden_eval_regression=hidden_regression,
                        extracted_skill_ids=extracted_skill_ids,
                        origin=mutation.origin,
                        nucleus_signal=nucleus_signal_to_dict(mutation_signal),
                    )
                    self._emit_event(
                        event_sink,
                        {
                            "event": "mutation_promoted",
                            "generation": generation,
                            "index": index,
                            "mutation_type": mutation.mutation_type,
                            "target": mutation.target,
                            "old_score": candidate_result.promotion_score,
                            "new_score": mutated_result.promotion_score,
                            "rationale": mutation.rationale,
                            "operator_type": operator.name,
                            "hidden_eval_regression": hidden_regression,
                        },
                    )
                    genome = mutated_genome
                    candidate_result = mutated_result
                    generation_terminal_result = mutated_result
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
                        f"stem_agent promoted {mutation.mutation_type}", [run_dir]
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
                    mutation_signal = bundle.scenario.signal_policy.build_nucleus_signal(
                        generation=generation,
                        mutation_type=mutation.mutation_type,
                        current_score=mutated_result.promotion_score,
                        previous_score=candidate_result.promotion_score,
                    )
                    signal_history.append(mutation_signal)
                    lineage.record_rolled_back_mutation(
                        generation,
                        mutation.mutation_type,
                        mutation.target,
                        candidate_result.promotion_score,
                        mutated_result.promotion_score,
                        "fitness did not improve enough for promotion",
                        operator_type=operator.name,
                        nucleus_signal=nucleus_signal_to_dict(mutation_signal),
                    )
                    self._emit_event(
                        event_sink,
                        {
                            "event": "mutation_rolled_back",
                            "generation": generation,
                            "index": index,
                            "mutation_type": mutation.mutation_type,
                            "target": mutation.target,
                            "old_score": candidate_result.promotion_score,
                            "new_score": mutated_result.promotion_score,
                            "reason": "fitness did not improve enough for promotion",
                            "operator_type": operator.name,
                        },
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
                        f"stem_agent rolled back {mutation.mutation_type}", [run_dir]
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

                sampled_parent_id, sampled_genome = archive.sample_parent(strategy="weighted")
                sampled_score = archive.score_for(sampled_parent_id)
                selected_parent_id = sampled_parent_id
                selected_genome = sampled_genome
                selected_score = sampled_score
                if best_result is not None and sampled_score < best_result.promotion_score:
                    selected_parent_id, selected_genome, selected_score = archive.best()
                genome = Genome.model_validate(selected_genome)
                current_parent_id = selected_parent_id
                self._active_parent_id = current_parent_id
                candidate_result = candidate_result.model_copy(
                    update={"score": selected_score, "promotion_score": selected_score}
                )
                lineage.record(
                    "archive_parent_sampled",
                    generation=generation,
                    sampled_parent_id=sampled_parent_id,
                    selected_parent_id=selected_parent_id,
                    candidate_genome_id=candidate_genome_id,
                    sampled_score=round(sampled_score, 4),
                    selected_score=round(selected_score, 4),
                )

            if not promoted_this_generation:
                patience_left -= 1
            if len(archive):
                archive_best_score = archive.best()[2]
                if last_archive_best_score is not None and archive_best_score - last_archive_best_score <= 0.01:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                last_archive_best_score = archive_best_score
                self._active_stagnation_count = stagnation_count
                if stagnation_count >= 3:
                    lineage.record(
                        "stagnation",
                        generation=generation,
                        stagnation_count=stagnation_count,
                    )
            observed_lift = generation_terminal_result.promotion_score - current_result.promotion_score
            for skill_id in injected_skill_ids:
                self.skill_library.record_observed_lift(
                    skill_id=skill_id,
                    run_id=run_id,
                    observed_score_lift=observed_lift,
                )
            evicted_skill_ids = self.skill_library.tick_probation(run_id=run_id)
            if evicted_skill_ids:
                lineage.record(
                    "skill_probation_evicted",
                    generation=generation,
                    skill_ids=evicted_skill_ids,
                )
            pending_awm_candidates = self.awm_bridge.analyze(
                run_id,
                {"run_dir": run_dir, "min_occurrences": 3},
            )
            if pending_awm_candidates:
                lineage.record(
                    "awm_candidates_queued",
                    generation=generation,
                    candidate_count=len(pending_awm_candidates),
                    candidate_ids=[
                        str(candidate.patch.get("candidate_id"))
                        for candidate in pending_awm_candidates
                    ],
                )
            saturation_atoms = self.skill_library.query_atoms(
                bundle.scenario.effective_domain_tags,
                current_run_id=run_id,
                current_scenario_name=bundle.scenario.name,
                top_k=5,
            )
            skill_saturation_available = bool(saturation_atoms)
            skill_saturation = (
                self.skill_library.saturation_score(bundle.scenario.effective_domain_tags)
                if skill_saturation_available
                else 0.0
            )
            tokens_used_this_generation = max(
                self._total_tokens_used(run_dir) - tokens_before_generation,
                0,
            )
            convergence.update(
                generation_number=generation,
                validation_score=self._validation_score(generation_terminal_result),
                hidden_eval_score=self._latest_hidden_score(run_dir, generation),
                tokens_used_this_generation=tokens_used_this_generation,
                skill_saturation_score=skill_saturation,
                skill_saturation_available=skill_saturation_available,
                zero_order_was_attempted=isinstance(operator, ZeroOrderMutation),
            )
            convergence_decision = convergence.evaluate()
            lineage.record(
                "convergence_decision",
                generation=generation,
                action=convergence_decision.action,
                stop=convergence_decision.stop,
                reason=convergence_decision.reason,
                plateau_detected=convergence_decision.plateau_detected,
            )
            self._emit_event(
                event_sink,
                {
                    "event": "convergence_decision",
                    "generation": generation,
                    "action": convergence_decision.action,
                    "stop": convergence_decision.stop,
                    "reason": convergence_decision.reason,
                    "plateau_detected": convergence_decision.plateau_detected,
                },
            )
            if convergence_decision.action == "force_zero_order":
                force_zero_order_next = True
            if convergence_decision.stop:
                stop_reason = convergence_decision.reason
                break
            self._emit_event(
                event_sink,
                {
                    "event": "generation_complete",
                    "generation": generation,
                    "best_score": best_result.promotion_score if best_result else 0.0,
                    "patience_left": patience_left,
                    "tokens_used": tokens_used_this_generation,
                },
            )
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
            git.commit_safe(f"stem_agent generation {generation} complete", [run_dir])
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
            event_sink=event_sink,
        )
        control.set_status("FROZEN", message=stop_reason)
        git.commit_safe("stem_agent final freeze", [run_dir])
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
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvolutionRunResult:
        frozen_path = run_dir / "frozen_genome.yaml"
        save_genome(best_genome, frozen_path)
        final_dir = run_dir / "final_evaluation"
        final_dir.mkdir(exist_ok=True)
        final_result = self._run_and_evaluate(
            best_genome,
            bundle,
            final_dir,
            "frozen",
            generation=generation,
            event_sink=event_sink,
        )
        self._write_json(final_dir / "eval_result.json", final_result.model_dump(mode="json"))
        self._write_json(run_dir / "run_metadata.json", self._run_metadata(final_result))
        lineage.record_freeze(
            generation,
            final_result.promotion_score,
            stop_reason,
        )
        self._emit_event(
            event_sink,
            {
                "event": "freeze",
                "generation": generation,
                "score": final_result.promotion_score,
                "reason": stop_reason,
            },
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
        archive: GenomeArchive | None = None,
        current_parent_id: str | None = None,
        stagnation_count: int | None = None,
    ) -> None:
        archive = archive or getattr(self, "_active_archive", None)
        if current_parent_id is None:
            current_parent_id = getattr(self, "_active_parent_id", None)
        if stagnation_count is None:
            stagnation_count = getattr(self, "_active_stagnation_count", 0)
        control.save_checkpoint(
            scenario_hash=scenario_hash,
            generation=generation,
            next_generation=next_generation,
            phase=phase,
            current_genome=genome,
            best_genome=best_genome,
            baseline_eval=baseline_result,
            best_eval=best_result,
            archive=archive.snapshot() if archive else None,
            stagnation_count=stagnation_count,
            patience_left=patience_left,
            stop_reason=stop_reason,
            current_parent_id=current_parent_id,
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
            git.commit_safe("stem_agent resume event", [run_dir])
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
            git.commit_safe("stem_agent safe pause checkpoint", [run_dir])
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
            git.commit_safe("stem_agent freeze-now", [run_dir])
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
            git.commit_safe("stem_agent abort checkpoint", [run_dir])
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
        *,
        generation: int | None = None,
        mutation_index: int | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvaluationResult:
        workspace_dir = generation_dir / f"{label}_workspace"
        self._emit_event(
            event_sink,
            {
                "event": "evaluation_start",
                "generation": generation,
                "label": label,
                "mutation_index": mutation_index,
                "genome_version": genome.genome_version,
                "workspace_dir": str(workspace_dir),
                "train_case_count": len(bundle.train_cases),
                "validation_case_count": len(bundle.validation_cases),
            },
        )
        run_root = self._run_root_for_artifact_dir(generation_dir)
        harness = self.harness_builder.materialize(
            genome,
            bundle.scenario,
            workspace_dir=workspace_dir,
        )
        train_runs = []
        for case in bundle.train_cases:
            self._emit_event(
                event_sink,
                {
                    "event": "case_start",
                    "generation": generation,
                    "label": label,
                    "split": "train",
                    "case_id": case.id,
                    "mutation_index": mutation_index,
                },
            )
            run = self.harness_runner.run_case(
                harness,
                case,
                trace_sink=self._case_trace_sink(
                    event_sink,
                    generation=generation,
                    label=label,
                    split="train",
                    case_id=case.id,
                    mutation_index=mutation_index,
                ),
                run_dir=run_root,
            )
            train_runs.append(run)
            self._emit_event(
                event_sink,
                {
                    "event": "case_complete",
                    "generation": generation,
                    "label": label,
                    "split": "train",
                    "case_id": case.id,
                    "mutation_index": mutation_index,
                    "output_summary": self._short_output_summary(run.final_output),
                },
            )
        validation_runs = []
        for case in bundle.validation_cases:
            self._emit_event(
                event_sink,
                {
                    "event": "case_start",
                    "generation": generation,
                    "label": label,
                    "split": "validation",
                    "case_id": case.id,
                    "mutation_index": mutation_index,
                },
            )
            run = self.harness_runner.run_case(
                harness,
                case,
                trace_sink=self._case_trace_sink(
                    event_sink,
                    generation=generation,
                    label=label,
                    split="validation",
                    case_id=case.id,
                    mutation_index=mutation_index,
                ),
                run_dir=run_root,
            )
            validation_runs.append(run)
            self._emit_event(
                event_sink,
                {
                    "event": "case_complete",
                    "generation": generation,
                    "label": label,
                    "split": "validation",
                    "case_id": case.id,
                    "mutation_index": mutation_index,
                    "output_summary": self._short_output_summary(run.final_output),
                },
            )
        self._write_runs(generation_dir, label, train_runs + validation_runs)
        self._emit_event(
            event_sink,
            {
                "event": "evaluation_artifacts_written",
                "generation": generation,
                "label": label,
                "mutation_index": mutation_index,
                "outputs_dir": str(generation_dir / "outputs"),
                "traces_path": str(generation_dir / f"{label}_traces.jsonl"),
            },
        )
        return self.guardian.evaluate_candidate(
            genome,
            bundle.scenario,
            train_runs,
            validation_runs,
            run_dir=run_root,
        )

    def _run_root_for_artifact_dir(self, artifact_dir: Path) -> Path:
        if artifact_dir.name.startswith("mutation_"):
            return artifact_dir.parent.parent
        return artifact_dir.parent

    def _hidden_case_count(self) -> int:
        path = self.guardian.hidden_cases_path
        if path is None or not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _run_hidden_evaluation(
        self,
        genome: Genome,
        bundle: ScenarioBundle,
        workspace_dir: Path,
        genome_id: str,
        *,
        generation: int,
        label: str,
        mutation_index: int | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> float:
        self._emit_event(
            event_sink,
            {
                "event": "hidden_evaluation_start",
                "generation": generation,
                "label": label,
                "genome_id": genome_id,
                "workspace_dir": str(workspace_dir),
                "mutation_index": mutation_index,
            },
        )
        harness = self.harness_builder.materialize(
            genome,
            bundle.scenario,
            workspace_dir=workspace_dir,
        )
        hidden_score = self.guardian._run_hidden_evaluation(
            harness,
            genome_id,
            generation=generation,
        )
        self._emit_event(
            event_sink,
            {
                "event": "hidden_evaluation_complete",
                "generation": generation,
                "label": label,
                "genome_id": genome_id,
                "workspace_dir": str(workspace_dir),
                "hidden_score": hidden_score,
                "mutation_index": mutation_index,
            },
        )
        return hidden_score

    def _case_trace_sink(
        self,
        event_sink: Callable[[dict[str, Any]], None] | None,
        *,
        generation: int | None,
        label: str,
        split: str,
        case_id: str,
        mutation_index: int | None,
    ) -> Callable[[dict[str, Any]], None] | None:
        if event_sink is None:
            return None

        def sink(trace: dict[str, Any]) -> None:
            event_sink(
                {
                    "event": "harness_trace",
                    "generation": generation,
                    "label": label,
                    "split": split,
                    "case_id": case_id,
                    "mutation_index": mutation_index,
                    "trace": trace,
                }
            )

        return sink

    def _emit_event(
        self,
        event_sink: Callable[[dict[str, Any]], None] | None,
        event: dict[str, Any],
    ) -> None:
        if event_sink:
            event_sink(event)

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

    def _short_output_summary(self, output: str, *, limit: int = 140) -> str:
        compact = " ".join(output.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _retrieve_reusable_skills(
        self,
        *,
        bundle: ScenarioBundle,
        run_id: str,
    ):
        skills = self.skill_library.query_atoms(
            bundle.scenario.effective_domain_tags,
            current_run_id=run_id,
            current_scenario_name=bundle.scenario.name,
            top_k=5,
        )
        for skill in skills:
            if skill.origin_scenario_name != bundle.scenario.name:
                self.skill_library.open_probation(
                    skill.id,
                    run_id=run_id,
                    scenario_name=bundle.scenario.name,
                )
        return skills

    def _extract_and_store_skills(
        self,
        *,
        old_genome: Genome,
        new_genome: Genome,
        mutation_type: str,
        score_before: float,
        score_after: float,
        run_id: str,
        scenario_name: str,
        generation: int,
        domain_tags: list[str],
    ) -> list[str]:
        atoms = self.skill_extractor.extract(
            old_genome=old_genome,
            new_genome=new_genome,
            mutation_type=mutation_type,
            score_before=score_before,
            score_after=score_after,
            run_id=run_id,
            scenario_name=scenario_name,
            generation_number=generation,
            scenario_domain_tags=domain_tags,
        )
        stored_ids: list[str] = []
        for atom in atoms:
            if self.skill_library.add_atom(atom):
                stored_ids.append(atom.id)
        return stored_ids

    def _validation_score(self, result: EvaluationResult) -> float:
        return float(result.validation_score if result.validation_score is not None else result.promotion_score)

    def _latest_hidden_score(self, run_dir: Path, generation: int) -> float | None:
        path = run_dir / "hidden_eval_log.jsonl"
        if not path.exists():
            return None
        latest: float | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("generation") == generation:
                latest = float(item.get("hidden_score", 0.0))
        return latest

    def _total_tokens_used(self, run_dir: Path) -> int:
        total = 0
        for path in [run_dir / "llm_calls.jsonl", *run_dir.glob("generation_*/llm_calls.jsonl")]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                total += int(item.get("tokens_used", 0))
        return total

    def _evaluation_summary(self, result: EvaluationResult) -> str:
        if result.failures:
            return "; ".join(result.failures[:3])
        return "Candidate satisfied visible requirements with acceptable complexity."

    def _run_metadata(self, final_result: EvaluationResult) -> dict:
        mode = "test double" if self.settings.test_mode else "openai-api"
        endpoint = self.model_client.endpoint
        responses_api_available = self.model_client.responses_api_available
        if self.settings.test_mode:
            responses_api_available = None
        elif endpoint == "chat_completions":
            responses_api_available = False
        chat_completions_fallback = (
            False
            if self.settings.test_mode
            else bool(self.model_client.fallback_used or endpoint == "chat_completions")
        )
        return {
            "run_mode": mode,
            "model": self.settings.model,
            "endpoint": endpoint,
            "test_mode": self.settings.test_mode,
            "fallback_used": self.model_client.fallback_used,
            "responses_api_available": responses_api_available,
            "chat_completions_fallback": chat_completions_fallback,
            "model_calls": self.model_client.model_calls,
            "structured_output_repairs": self.model_client.structured_output_repairs,
            "total_estimated_cost": final_result.cost_estimate,
        }

    def _load_signal_history(self, lineage: LineageLog) -> list[NucleusSignal]:
        signals: list[NucleusSignal] = []
        for event in lineage.events:
            signal = event.get("nucleus_signal")
            if isinstance(signal, dict):
                try:
                    signals.append(NucleusSignal(**signal))
                except TypeError:
                    continue
        return signals

    def _gsm8k_baseline_genome(self, scenario_name: str) -> Genome:
        return Genome.model_validate(
            {
                "genome_version": 0,
                "name": "gsm8k_zero_shot_cot_seed",
                "scenario_name": scenario_name,
                "task_diagnosis": {},
                "roles": [
                    {
                        "name": "solver",
                        "description": "Zero-shot chain-of-thought math solver.",
                        "instructions": "Solve the math problem. Show your work.",
                        "allowed_tools": ["call_model"],
                    }
                ],
                "workflow": [
                    {
                        "id": "solve",
                        "role": "solver",
                        "action": "Solve the math problem. Show your work.",
                        "input_from": [],
                        "output_key": "final_output",
                    }
                ],
                "tools": {"builtin": [], "generated": []},
                "memory": {},
                "quality_gates": [],
                "self_evaluation": {"rubric": ""},
                "retry_policy": {"max_attempts": 1, "revise_on_failure": False},
                "stop_rule": {},
                "environment": {
                    "workspace_layout": [],
                    "required_artifacts": [],
                    "artifact_purpose": {},
                    "file_templates": {},
                    "cleanup_policy": "keep_run_artifacts",
                },
            }
        )
