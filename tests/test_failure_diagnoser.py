import pytest

from stem_agent.harness.failure_diagnoser import (
    DIAGNOSIS_PROMPT,
    FailureDiagnoser,
    _extract_output,
    _truncate,
)
from stem_agent.harness.runner import HarnessRunResult
from stem_agent.kernel.evaluator import CaseEvaluation


class TestFailureDiagnoser:
    """Unit tests for the FailureDiagnoser."""

    def test_disabled_returns_empty(self):
        d = FailureDiagnoser(enabled=False)
        result = d.diagnose([], [])
        assert result == []

    def test_no_model_client_returns_empty(self):
        d = FailureDiagnoser(enabled=True)
        result = d.diagnose([], [])
        assert result == []

    def test_resolved_case_returns_passed(self):
        d = FailureDiagnoser(enabled=True)
        # Mock model_client
        d._model_client = _FakeModelClient({"reason": "PASSED"})
        case = CaseEvaluation(
            case_id="test-001",
            score=1.0,
            metrics={},
            eval_detail={"resolved": True, "patch_applies": True},
        )
        run = HarnessRunResult(case_id="test-001", final_output="ok", outputs={})
        result = d.diagnose([case], [run])
        assert len(result) == 1
        assert result[0] == "PASSED"

    def test_failed_case_diagnoses(self):
        d = FailureDiagnoser(enabled=True)
        d._model_client = _FakeModelClient(
            "locator never called read_file, hallucinated line numbers"
        )
        case = CaseEvaluation(
            case_id="test-002",
            score=0.3,
            metrics={},
            failures=["Patch doesn't apply"],
            eval_detail={
                "resolved": False,
                "patch_applies": False,
                "fail_to_pass": {"test_a": "FAILED"},
                "pass_to_pass": {},
                "error": None,
            },
        )
        run = HarnessRunResult(
            case_id="test-002",
            final_output="--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n # Some code...\n",
            outputs={
                "code_locations": "I found the bug at line 100 (estimated). # ... some code ...",
                "final_output": "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n # Some code...\n",
                "verification_result": "Patch FAILED to apply",
            },
        )
        result = d.diagnose([case], [run])
        assert len(result) == 1
        assert "hallucinated" in result[0].lower() or "line" in result[0].lower()

    def test_no_matching_run(self):
        d = FailureDiagnoser(enabled=True)
        d._model_client = _FakeModelClient("should not be called")
        case = CaseEvaluation(
            case_id="missing",
            score=0.0,
            metrics={},
            eval_detail={},
        )
        run = HarnessRunResult(case_id="different", final_output="ok", outputs={})
        result = d.diagnose([case], [run])
        assert result == ["no harness data"]

    def test_trace_roundtrip(self):
        """Verify that runs written via model_dump can be read back."""
        original = HarnessRunResult(
            case_id="case-1",
            final_output="hello",
            outputs={"code_locations": "locator output here", "final_output": "patch here"},
            cost_estimate=0.1,
        )
        data = original.model_dump(mode="json")
        restored = HarnessRunResult.model_validate(data)
        assert restored.case_id == "case-1"
        assert restored.outputs["code_locations"] == "locator output here"
        assert restored.outputs["final_output"] == "patch here"


class TestExtractOutput:
    def test_refined_preferred(self):
        outputs = {
            "_refined_code_locations": '{"source_file": "a.py"}',
            "code_locations": "raw locator output",
        }
        result = _extract_output(outputs, "code_locations", "locate")
        assert result == outputs["_refined_code_locations"]

    def test_fallback_to_raw(self):
        outputs = {
            "code_locations": "raw locator output",
        }
        result = _extract_output(outputs, "code_locations", "locate")
        assert result == "raw locator output"

    def test_multiple_keys(self):
        outputs = {
            "locate": "locate output",
        }
        result = _extract_output(outputs, "code_locations", "locate")
        assert result == "locate output"

    def test_missing_returns_placeholder(self):
        result = _extract_output({}, "code_locations")
        assert result == "(no output)"


class TestTruncate:
    def test_short_no_truncation(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_truncation(self):
        text = "x" * 3000
        result = _truncate(text, 2000)
        assert len(result) <= 2100  # some overhead for truncation message
        assert "[truncated" in result

    def test_exact_at_limit(self):
        text = "x" * 2000
        result = _truncate(text, 2000)
        assert result == text


class _FakeModelClient:
    """Fake model client for testing without real API calls."""

    def __init__(self, response):
        self._response = response

    def call(self, prompt, *, temperature=None, role=None, **kwargs):
        return self._response

    def configure_run(self, run_dir=None):
        pass
