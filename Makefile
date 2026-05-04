.PHONY: test evolve-demo inspect-demo compare-demo

test:
	python3 -m pytest

evolve-demo:
	PYTHONPATH=src python3 -m stemos.cli evolve scenarios/tiny_task_operator --run-id demo_001

inspect-demo:
	PYTHONPATH=src python3 -m stemos.cli inspect runs/demo_001

compare-demo:
	PYTHONPATH=src python3 -m stemos.cli compare runs/demo_001
