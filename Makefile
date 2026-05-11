.PHONY: test evolve-demo inspect-demo compare-demo init-gsm8k-demo evolve-gsm8k-demo init-gsm8k-full evolve-gsm8k-full

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
