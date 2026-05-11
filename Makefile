.PHONY: test evolve-demo inspect-demo compare-demo init-gsm8k-demo evolve-gsm8k-demo init-gsm8k-full evolve-gsm8k-full list-benchmarks evolve-security-review-demo evolve-research-synthesis-demo evolve-release-qa-triage-demo evolve-robustness-demo

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

evolve-security-review-demo:
	stem_agent evolve scenarios/security_review_demo --run-id security_review_demo_001

evolve-research-synthesis-demo:
	stem_agent evolve scenarios/research_synthesis_demo --run-id research_synthesis_demo_001

evolve-release-qa-triage-demo:
	stem_agent evolve scenarios/release_qa_triage_demo --run-id release_qa_triage_demo_001

evolve-robustness-demo:
	stem_agent evolve scenarios/security_review_demo --run-id security_review_demo_001
	stem_agent evolve scenarios/research_synthesis_demo --run-id research_synthesis_demo_001
	stem_agent evolve scenarios/release_qa_triage_demo --run-id release_qa_triage_demo_001
