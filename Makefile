.PHONY: test evolve-demo inspect-demo compare-demo init-gsm8k-demo evolve-gsm8k-demo init-gsm8k-full evolve-gsm8k-full list-benchmarks evolve-security-review-mini evolve-research-synthesis-mini evolve-release-qa-triage-mini evolve-robustness-mini

test:
	python3 -m pytest

evolve-demo:
	stem_agent evolve scenarios/tiny_task_operator --run-id demo_001

inspect-demo:
	stem_agent inspect runs/demo_001

compare-demo:
	stem_agent compare runs/demo_001

init-gsm8k-demo:
	stem_agent init-benchmark gsm8k_demo

evolve-gsm8k-demo:
	stem_agent evolve scenarios/gsm8k_demo --run-id gsm8k_demo_001

init-gsm8k-full:
	stem_agent init-benchmark gsm8k_full

evolve-gsm8k-full:
	stem_agent evolve scenarios/gsm8k_full --run-id gsm8k_full_001

list-benchmarks:
	stem_agent list-benchmarks

evolve-security-review-mini:
	stem_agent evolve scenarios/security_review_mini --run-id security_review_mini_001

evolve-research-synthesis-mini:
	stem_agent evolve scenarios/research_synthesis_mini --run-id research_synthesis_mini_001

evolve-release-qa-triage-mini:
	stem_agent evolve scenarios/release_qa_triage_mini --run-id release_qa_triage_mini_001

evolve-robustness-mini:
	stem_agent evolve scenarios/security_review_mini --run-id security_review_mini_001
	stem_agent evolve scenarios/research_synthesis_mini --run-id research_synthesis_mini_001
	stem_agent evolve scenarios/release_qa_triage_mini --run-id release_qa_triage_mini_001
