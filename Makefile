.PHONY: test evolve-demo inspect-demo compare-demo

test:
	python3 -m pytest

evolve-demo:
	stem_agent evolve scenarios/tiny_task_operator --run-id demo_001

inspect-demo:
	stem_agent inspect runs/demo_001

compare-demo:
	stem_agent compare runs/demo_001
