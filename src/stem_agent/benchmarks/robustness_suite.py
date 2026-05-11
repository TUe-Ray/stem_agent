from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    task_class: str
    path: str
    focus: str
    run_id: str

    @property
    def evolve_command(self) -> str:
        return f"stem_agent evolve {self.path} --run-id {self.run_id}"


ROBUSTNESS_BENCHMARKS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        name="security_review_demo",
        task_class="security_review_triage",
        path="scenarios/security_review_demo",
        focus="Defensive review with threat model, severity ranking, remediation, and verification.",
        run_id="security_review_demo_001",
    ),
    BenchmarkScenario(
        name="research_synthesis_demo",
        task_class="deep_research_synthesis",
        path="scenarios/research_synthesis_demo",
        focus="Grounded synthesis from supplied notes with evidence, uncertainty, and recommendation.",
        run_id="research_synthesis_demo_001",
    ),
    BenchmarkScenario(
        name="release_qa_triage_demo",
        task_class="release_quality_triage",
        path="scenarios/release_qa_triage_demo",
        focus="Release readiness triage with test plan, risk register, go/no-go call, and rollback.",
        run_id="release_qa_triage_demo_001",
    ),
)


__all__ = ["BenchmarkScenario", "ROBUSTNESS_BENCHMARKS"]
