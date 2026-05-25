"""Tests for SWE-bench evaluator: _extract_diff and mock criteria methods."""
import os
import pytest

from stem_agent.kernel.evaluator import GuardianFitnessEvaluator


@pytest.fixture
def evaluator():
    """Create an uninitialized evaluator instance for testing static/helper methods."""
    e = GuardianFitnessEvaluator.__new__(GuardianFitnessEvaluator)
    return e


class TestExtractDiff:
    def test_markdown_fenced_diff(self, evaluator):
        """Extract diff from markdown-fenced block."""
        output = (
            "Here is my analysis:\n"
            "```diff\n"
            "--- a/views.py\n"
            "+++ b/views.py\n"
            "@@ -42,6 +42,10 @@ def login(request):\n"
            "+    if not username:\n"
            "+        return 400\n"
            "```\n"
            "That should fix it."
        )
        diff = evaluator._extract_diff(output)
        assert diff.startswith("--- a/views.py"), f"Got: {diff[:50]}"
        assert "@@" in diff
        assert "```" not in diff

    def test_raw_diff_unchanged(self, evaluator):
        """Raw diff passes through unchanged."""
        output = "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-foo\n+bar\n"
        diff = evaluator._extract_diff(output)
        assert diff == output.strip()

    def test_diff_git_header(self, evaluator):
        """diff --git header found in prose."""
        output = (
            "## Patch\n\n"
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1 +1 @@\n"
        )
        diff = evaluator._extract_diff(output)
        assert diff.startswith("diff --git")

    def test_no_diff_returns_original(self, evaluator):
        """Non-diff text is returned as-is."""
        output = "Just some analysis text. No patch here."
        diff = evaluator._extract_diff(output)
        assert "analysis" in diff

    def test_trailing_diff_markers(self, evaluator):
        """Find ---/+++/@@ at end of prose (Case 3 fallback)."""
        output = "Analysis complete.\n+++ b/file.py\n@@ -5,3 +5,8 @@\n+def new_test():\n+    pass\n"
        diff = evaluator._extract_diff(output)
        assert "+++ b/file.py" in diff
        assert "@@" in diff

    def test_empty_output(self, evaluator):
        """Empty string is handled."""
        assert evaluator._extract_diff("") == ""
        assert evaluator._extract_diff("   ") == ""


class TestMockPatchAppliesScore:
    def test_real_mode_with_diff(self, evaluator, monkeypatch):
        """In real mode (no workspace), prose-wrapped diff gets heuristic 0.5."""
        output = (
            "Here's the fix:\n"
            "```diff\n"
            "--- a/views.py\n"
            "+++ b/views.py\n"
            "@@ -42,6 +42,10 @@ def login(request):\n"
            "+    if not username:\n"
            "+        return HttpResponseBadRequest()\n"
            "```\n"
        )
        score = evaluator._patch_applies_score(None, output, None)
        assert score == 0.5, f"Expected 0.5 (heuristic fallback), got {score}"

    def test_real_mode_no_diff(self, evaluator, monkeypatch):
        """Prose without diff header scores 0."""
        output = "I looked at the code and everything seems fine."
        score = evaluator._patch_applies_score(None, output, None)
        assert score == 0.0, f"Expected 0.0, got {score}"

    def test_real_mode_partial_diff(self, evaluator, monkeypatch):
        """Diff header without hunks gets 0.0 (no hunks)."""
        output = "--- a/file.py\n+++ b/file.py\n"
        score = evaluator._patch_applies_score(None, output, None)
        assert score == 0.0, f"Expected 0.0 (no hunks), got {score}"


class TestMockSwebenchDockerEvalScore:
    def test_mock_mode_full_diff(self, evaluator, monkeypatch):
        """Full diff with header + hunks gets 0.8."""
        monkeypatch.setenv("STEM_AGENT_SWEBENCH_MOCK", "1")
        output = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        score = evaluator._swebench_docker_eval_score(None, output, None)
        assert score == 0.8, f"Expected 0.8, got {score}"

    def test_mock_mode_empty_output(self, evaluator, monkeypatch):
        """Empty output scores 0."""
        monkeypatch.setenv("STEM_AGENT_SWEBENCH_MOCK", "1")
        score = evaluator._swebench_docker_eval_score(None, "", None)
        assert score == 0.0


class TestRegexCheckMultiline:
    def test_multiline_anchors_match(self, evaluator):
        """With MULTILINE flag, ^ anchors match mid-string lines."""
        output = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
        pattern = r"^\+{3}.*$|^---.*$|^@@.*@@$"
        import re
        assert re.search(pattern, output, re.DOTALL | re.MULTILINE | re.IGNORECASE) is not None
