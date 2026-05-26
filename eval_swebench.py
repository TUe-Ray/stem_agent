"""Manual SWE-bench evaluation script — runs real pytest eval on agent outputs.

Usage:
    python3 eval_swebench.py runs/swe_v11
"""

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stem_agent.benchmarks.swebench import evaluate_patch_light


def load_frozen_traces(run_dir: Path) -> list[dict]:
    traces_path = run_dir / "final_evaluation" / "frozen_traces.jsonl"
    if not traces_path.exists():
        print(f"❌ No frozen traces at {traces_path}")
        return []
    return [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]


def eval_run(run_dir: Path) -> dict:
    """Run real SWE-bench evaluation on all cases in a run."""
    traces = load_frozen_traces(run_dir)
    if not traces:
        return {}

    results = {"run": str(run_dir.name), "cases": [], "summary": {}}

    for trace in traces:
        case_id = trace.get("case_id", "unknown")
        case_input = trace.get("case_input", {})
        final_output = trace.get("final_output", "")

        print(f"\n{'='*60}")
        print(f"📋 Case: {case_id}")
        print(f"   Repo: {case_input.get('repo')}")
        print(f"   Patch length: {len(final_output)} chars")

        if not final_output.strip():
            print("   ⚠️  Empty output — skipping")
            results["cases"].append({"case_id": case_id, "error": "empty output"})
            continue

        # Run real evaluation
        work_dir = Path(f"/tmp/swebench_eval/{run_dir.name}/{case_id}")
        work_dir.parent.mkdir(parents=True, exist_ok=True)

        print(f"   Running pytest eval...")
        eval_result = evaluate_patch_light(
            instance={
                "instance_id": case_id,
                "repo": case_input.get("repo", ""),
                "base_commit": case_input.get("base_commit", ""),
                "problem_statement": case_input.get("problem_statement", ""),
                "test_patch": case_input.get("test_patch", ""),
                "hints_text": case_input.get("hints_text", ""),
                "FAIL_TO_PASS": _parse_json_field(case_input.get("FAIL_TO_PASS", [])),
                "PASS_TO_PASS": _parse_json_field(case_input.get("PASS_TO_PASS", [])),
            },
            predicted_patch=final_output,
            work_dir=work_dir,
            timeout=300,
        )

        case_result = {
            "case_id": case_id,
            "resolved": eval_result.get("resolved", False),
            "patch_applies": eval_result.get("patch_applies", False),
            "score": eval_result.get("score", 0.0),
            "fail_to_pass": eval_result.get("fail_to_pass", {}),
            "pass_to_pass": eval_result.get("pass_to_pass", {}),
            "error": eval_result.get("error"),
        }
        results["cases"].append(case_result)

        status = "✅ RESOLVED" if case_result["resolved"] else (
            "🔧 PATCH OK" if case_result["patch_applies"] else "❌ NO APPLY"
        )
        print(f"   {status} | score={case_result['score']:.2f}")
        if case_result["error"]:
            print(f"   Error: {case_result['error'][:200]}")

    # Summary
    total = len(results["cases"])
    resolved = sum(1 for c in results["cases"] if c.get("resolved"))
    applied = sum(1 for c in results["cases"] if c.get("patch_applies"))
    results["summary"] = {
        "total": total,
        "resolved": resolved,
        "patch_applied": applied,
        "resolve_rate": f"{resolved}/{total} ({resolved/total*100:.1f}%)" if total else "N/A",
        "avg_score": sum(c.get("score", 0) for c in results["cases"]) / max(1, total),
    }

    return results


def _parse_json_field(value):
    """Parse a field that might be a JSON string from HF datasets."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    if isinstance(value, list):
        return value
    return []


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/swe_v10")
    print(f"🔬 Evaluating: {run_dir}")
    results = eval_run(run_dir)

    # Save results
    out_path = run_dir / "swebench_real_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"📊 Summary: {results['summary']}")
    print(f"💾 Saved to: {out_path}")
