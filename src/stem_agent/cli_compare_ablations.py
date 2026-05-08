from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def compare_ablations(run_ids: list[str]) -> str:
    run_paths = [Path(item) for item in run_ids]
    reports = {path.name: _run_metrics(path) for path in run_paths}
    metrics = [
        "Generations to score>0.7",
        "Final validation score",
        "Final hidden eval score",
        "Overfitting gap",
        "Mutation type entropy",
        "Stagnation events",
    ]
    header = ["| Metric |", "|---|"]
    for path in run_paths:
        header[0] += f" {path.name} |"
        header[1] += "---:|"
    rows = []
    for metric in metrics:
        row = f"| {metric} |"
        for path in run_paths:
            row += f" {reports[path.name].get(metric, 'N/A')} |"
        rows.append(row)
    content = "\n".join(["# stem_agent Ablation Comparison", "", header[0], header[1], *rows, ""])
    output_path = Path("runs") / "ablation_comparison.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)


def _run_metrics(run_dir: Path) -> dict[str, str]:
    lineage = _read_jsonl(run_dir / "lineage.jsonl")
    generation_scores = [
        (int(event.get("generation", 0)), float(event.get("score", 0.0)))
        for event in lineage
        if event.get("event") == "evaluation"
    ]
    final_eval = _read_json(run_dir / "final_evaluation" / "eval_result.json")
    hidden_rows = _read_jsonl(run_dir / "hidden_eval_log.jsonl")
    final_validation = _final_validation_score(final_eval, generation_scores)
    hidden_score = _final_hidden_score(hidden_rows)
    mutation_types = [
        str(event.get("mutation_type"))
        for event in lineage
        if event.get("event") in {"mutation_promoted", "mutation_rejected", "mutation_rolled_back"}
        and event.get("mutation_type")
    ]
    return {
        "Generations to score>0.7": _generations_to_threshold(generation_scores, 0.7),
        "Final validation score": _fmt(final_validation),
        "Final hidden eval score": _fmt(hidden_score),
        "Overfitting gap": _fmt(final_validation - hidden_score if final_validation is not None and hidden_score is not None else None),
        "Mutation type entropy": _entropy_bits(mutation_types),
        "Stagnation events": str(
            sum(
                1
                for event in lineage
                if event.get("event") == "stagnation" and int(event.get("stagnation_count", 0)) >= 3
            )
        ),
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _generations_to_threshold(generation_scores: list[tuple[int, float]], threshold: float) -> str:
    for generation, score in sorted(generation_scores):
        if score >= threshold:
            return str(generation)
    if generation_scores:
        best = max(score for _, score in generation_scores)
        return f"N/A (best {_fmt(best)})"
    return "N/A"


def _final_validation_score(final_eval: dict, generation_scores: list[tuple[int, float]]) -> float | None:
    if final_eval:
        value = final_eval.get("validation_score")
        if value is None:
            value = final_eval.get("promotion_score", final_eval.get("score"))
        if value is not None:
            return float(value)
    if generation_scores:
        return float(generation_scores[-1][1])
    return None


def _final_hidden_score(hidden_rows: list[dict]) -> float | None:
    if not hidden_rows:
        return None
    return float(hidden_rows[-1].get("hidden_score", 0.0))


def _entropy_bits(values: list[str]) -> str:
    if not values:
        return "N/A"
    counts = Counter(values)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return f"{entropy:.2f} bits"


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"
