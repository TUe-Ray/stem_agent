from io import StringIO

from stem_agent.cli import (
    _TrainingTerminalReporter,
    _format_execution_transcript,
    _format_training_detail_event,
    _format_training_transcript_event,
)


def test_format_execution_transcript_reads_like_agent_dialogue():
    transcript = _format_execution_transcript(
        [
            {
                "event": "environment_materialized",
                "artifacts": ["artifacts/final_output.md"],
            },
            {
                "event": "workflow_step",
                "role": "Founder",
                "step_id": "solve_task",
                "step_action": "Produce the final output.",
                "agent_observation": {
                    "goal": "Produce the final output.",
                    "available_context": "case_input=user_request: Plan my day.",
                    "output_summary": "## Summary Make a focused plan.",
                },
            },
            {
                "event": "quality_gate",
                "gate": "required_sections_gate",
                "passed": True,
                "missing": [],
            },
        ]
    )

    assert "[Harness / environment]" in transcript
    assert "[Founder / solve_task]" in transcript
    assert "I can see: case_input=user_request: Plan my day." in transcript
    assert "I produced: ## Summary Make a focused plan." in transcript
    assert "[Guardian / required_sections_gate]" in transcript
    assert "not hidden chain-of-thought" in transcript


def test_format_training_transcript_event_shows_decisions():
    proposed = _format_training_transcript_event(
        {
            "event": "mutation_proposed",
            "generation": 2,
            "index": 0,
            "mutation_type": "add_workflow_step",
            "target": "workflow",
            "rationale": "Add review before final output.",
            "expected_improvement": "Improve requirement coverage.",
            "risk": "Adds complexity.",
        }
    )
    promoted = _format_training_transcript_event(
        {
            "event": "mutation_promoted",
            "generation": 2,
            "index": 0,
            "mutation_type": "add_workflow_step",
            "target": "workflow",
            "old_score": 0.5,
            "new_score": 0.7,
            "rationale": "Review improved final output.",
        }
    )

    assert "[Training / generation 2 / Nucleus proposes mutation 0]" in proposed
    assert "Why: Add review before final output." in proposed
    assert "[Training / generation 2 / Guardian promotes mutation 0]" in promoted
    assert "Score: 0.5000 -> 0.7000" in promoted


def test_format_training_transcript_event_wraps_harness_trace():
    transcript = _format_training_transcript_event(
        {
            "event": "harness_trace",
            "generation": 1,
            "label": "candidate",
            "split": "train",
            "case_id": "train_001",
            "mutation_index": 0,
            "trace": {
                "event": "workflow_step",
                "role": "Founder",
                "step_id": "solve_task",
                "step_action": "Produce the final output.",
                "agent_observation": {
                    "goal": "Produce the final output.",
                    "available_context": "case_input=user_request: Plan my day.",
                    "output_summary": "## Summary Make a focused plan.",
                },
            },
        }
    )

    assert "[Training / generation 1 / candidate / train:train_001]" in transcript
    assert "Mutation candidate: 0" in transcript
    assert "[Founder / solve_task]" in transcript


def test_format_training_detail_event_shows_run_overview():
    overview = _format_training_detail_event(
        {
            "event": "evolution_start",
            "run_id": "demo_001",
            "run_dir": "runs/demo_001",
            "scenario": "toy_structured_answer",
            "scenario_path": "scenarios/toy_structured_answer/scenario.yaml",
            "task_class": "structured_answer",
            "resume": False,
            "model": "gpt-4.1-mini",
            "endpoint": "chat_completions",
            "train_case_count": 2,
            "validation_case_count": 1,
            "hidden_case_count": 0,
            "max_generations": 5,
            "max_mutations_per_generation": 4,
            "patience": 2,
            "min_delta": 0.03,
            "max_cost_usd": 2.0,
        }
    )

    assert "=== stem_agent evolution ===" in overview
    assert "Scenario: toy_structured_answer (structured_answer)" in overview
    assert "Run directory: runs/demo_001" in overview
    assert "train cases: 2" in overview
    assert "hidden evaluation: not configured" in overview


def test_training_reporter_prints_progress_and_details_without_ansi():
    stream = StringIO()
    reporter = _TrainingTerminalReporter(
        3,
        stream=stream,
        force_ansi=False,
    )

    reporter.on_event({"event": "generation_start", "generation": 1, "genome_version": "v1"})
    reporter.on_event(
        {
            "event": "evaluation_complete",
            "generation": 1,
            "label": "current",
            "score": 0.42,
            "train_score": 0.4,
            "validation_score": 0.45,
        }
    )

    output = stream.getvalue()
    assert output.splitlines()[0].startswith("Training progress")
    assert "=== Generation 1 ===" in output
    assert "Evaluation complete: current" in output
    assert "score 0.4200" in output
