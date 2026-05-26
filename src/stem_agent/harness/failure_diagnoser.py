"""Failure Diagnoser — cheap LLM-based per-case failure reason extractor.

After evaluation, looks at per-case eval_details + locator/patcher output traces
and produces compact per-case failure reasons for Nucleus, so it can propose
targeted fixes instead of blind guesses.

Pattern: follows ContextRefiner's approach of using a cheap LLM call to extract
structured insights from verbose agent outputs between workflow steps.
"""

from __future__ import annotations

import json
from typing import Any

from stem_agent.harness.runner import HarnessRunResult
from stem_agent.kernel.evaluator import CaseEvaluation

DIAGNOSIS_PROMPT = """\
You are diagnosing why an AI coding agent failed to fix a bug.
Given the evaluation result and agent outputs below, write ONE sentence
explaining the most likely root cause of failure.

EVALUATION:
- case_id: {case_id}
- score: {score}
- resolved: {resolved}
- patch_applies: {patch_applies}
- fail_to_pass: {fail_to_pass}
- pass_to_pass: {pass_to_pass}
- error: {error}
- failures: {failures}

LOCATOR OUTPUT (what the agent found in the codebase):
{locator_output}

PATCHER OUTPUT (the patch the agent produced):
{patcher_output}

VERIFIER OUTPUT (patch application result):
{verifier_output}

Write ONE short sentence (max 200 chars) diagnosing WHY this case failed.
Focus on root causes like:
- "locator hallucinated line numbers, never called read_file"
- "patch had placeholder comments instead of real code"
- "patcher targeted wrong source file"
- "patch used estimated/nonexistent line numbers"
- "locator output was empty/missing"
- "patcher produced malformed diff headers"
- "patch context lines don't match actual source"
- "agent output was truncated or incomplete"
- "verifier found patch doesn't apply cleanly"

If the case PASSED (resolved=true, score > 0.9), respond with exactly "PASSED".

Output ONLY the sentence, no explanation, no markdown, no quotes.
"""

# ── Output step keys to extract for diagnosis ──
_LOCATOR_KEYS = ("code_locations", "locate")
_PATCHER_KEYS = ("final_output", "self_verifying_patch", "patch", "solve")
_VERIFIER_KEYS = ("verification_result", "verify_final", "verify")


class FailureDiagnoser:
    """Cheap LLM-based per-case failure reason extractor.

    Takes eval_details + harness run output traces and produces compact
    per-case failure reasons for NucleusSignal.
    """

    def __init__(self, model_client=None, enabled: bool = True):
        self._model_client = model_client  # set later via configure()
        self.enabled = enabled

    def configure(self, model_client) -> None:
        self._model_client = model_client

    def diagnose(
        self,
        case_evaluations: list[CaseEvaluation],
        harness_runs: list[HarnessRunResult],
    ) -> list[str]:
        """Return one failure reason string per case.

        Args:
            case_evaluations: Per-case evaluation results from evaluator.
            harness_runs: Per-case harness runs with raw step outputs.

        Returns:
            List of failure reason strings, one per case. Empty list
            if disabled or no model client.
        """
        if not self.enabled or self._model_client is None:
            return []

        # Build a lookup from case_id → harness run
        run_by_id: dict[str, HarnessRunResult] = {}
        for run in harness_runs:
            run_by_id[run.case_id] = run

        reasons: list[str] = []
        for case_eval in case_evaluations:
            run = run_by_id.get(case_eval.case_id)
            if run is None:
                reasons.append("no harness data")
                continue

            reason = self._diagnose_single(case_eval, run)
            reasons.append(reason)

        return reasons

    def _diagnose_single(
        self,
        case_eval: CaseEvaluation,
        run: HarnessRunResult,
    ) -> str:
        """Diagnose a single case failure."""
        # Quick pass: resolved cases
        if case_eval.eval_detail.get("resolved") and case_eval.score > 0.9:
            return "PASSED"

        # Extract step outputs
        locator_output = _extract_output(run.outputs, *_LOCATOR_KEYS)
        patcher_output = _extract_output(run.outputs, *_PATCHER_KEYS)
        verifier_output = _extract_output(run.outputs, *_VERIFIER_KEYS)

        raw_eval = case_eval.eval_detail
        prompt = DIAGNOSIS_PROMPT.format(
            case_id=case_eval.case_id,
            score=round(case_eval.score, 3),
            resolved=raw_eval.get("resolved", False),
            patch_applies=raw_eval.get("patch_applies", False),
            fail_to_pass=_fmt_test_dict(raw_eval.get("fail_to_pass", {})),
            pass_to_pass=_fmt_test_dict(raw_eval.get("pass_to_pass", {})),
            error=raw_eval.get("error") or "none",
            failures=", ".join(case_eval.failures) if case_eval.failures else "none",
            locator_output=_truncate(locator_output, 2000),
            patcher_output=_truncate(patcher_output, 2000),
            verifier_output=_truncate(verifier_output, 800),
        )

        try:
            # ── Langfuse trace for diagnosis step ──
            _langfuse_log_diagnosis(case_eval.case_id, prompt)

            result = self._model_client.call(
                prompt,
                temperature=0.0,
                role="failure_diagnoser",
            )
            if isinstance(result, dict):
                return str(result.get("reason", result))
            return str(result).strip().replace('"', "")
        except Exception as exc:
            return f"diagnosis error: {exc!s}"[:200]


def _extract_output(outputs: dict[str, str], *keys: str) -> str:
    """Extract the first matching output from harness outputs dict.

    Checks refined versions first (_refined_ prefix), then raw outputs.
    """
    for key in keys:
        refined_key = f"_refined_{key}"
        if refined_key in outputs:
            return outputs[refined_key]
    for key in keys:
        if key in outputs:
            return outputs[key]
    return "(no output)"


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text for prompt fitting."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars // 2
    return text[:head] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-tail:]


def _fmt_test_dict(d: dict[str, str | list]) -> str:
    """Format a test dict for prompt display."""
    if not d:
        return "none"
    items = []
    for k, v in d.items():
        if isinstance(v, list):
            items.append(f"{k}: {len(v)} tests")
        else:
            items.append(f"{k}: {v}")
    return ", ".join(items[:5])


def _langfuse_log_diagnosis(case_id: str, prompt: str) -> None:
    """Log a diagnosis call as a Langfuse tool_call event."""
    try:
        from stem_agent.observability.langfuse_tracer import log_tool_call
        log_tool_call(
            tool_name="failure_diagnoser",
            args={"case_id": case_id},
            result_summary=f"Diagnosing case {case_id} ({len(prompt)} chars prompt)",
            success=True,
            duration_ms=0.0,
        )
    except Exception:
        pass
