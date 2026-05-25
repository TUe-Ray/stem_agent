"""Phase D: Requirement semantic matching via synonyms (test_requirements_semantic)."""
import pytest

from stem_agent.kernel.requirements import requirement_satisfied, requirement_section_satisfied


class TestRequirementSectionSatisfied:
    def test_summary_keyword_matches(self):
        """'Include a summary' is satisfied when 'summary' is in output."""
        assert requirement_section_satisfied(
            "Must include a summary", "## Summary\nThis is the output."
        )

    def test_steps_keyword_matches_numbered_list(self):
        """'List steps' is satisfied by '1.' pattern."""
        assert requirement_section_satisfied(
            "Must list steps", "1. First step\n2. Second step"
        )

    def test_final_answer_keyword_matches(self):
        """'Include final answer' is satisfied by 'final answer' in output."""
        assert requirement_section_satisfied(
            "Include final answer", "## Final Answer\n42"
        )

    def test_unrecognized_section_returns_none(self):
        """Unknown section type returns None (fall through to word match)."""
        result = requirement_section_satisfied(
            "Must be well-written", "Some text here"
        )
        assert result is None


class TestRequirementSatisfied:
    def test_majority_words_match(self):
        """Requirement satisfied when majority of significant words appear."""
        assert requirement_satisfied(
            "Must include concrete implementation steps",
            "Here are concrete implementation steps for the project.",
        )

    def test_insufficient_word_overlap_fails(self):
        """Requirement fails when too few significant words appear."""
        assert not requirement_satisfied(
            "Must include detailed performance benchmarks with graphs",
            "This is a simple answer.",
        )

    def test_edge_case_no_long_words(self):
        """Requirements with no ≥4-letter words are trivially satisfied if no words to match."""
        # "Be nice" → words = ["nice"] → "nice" is in "Be nice" but not in "Any output here."
        # So this actually requires word overlap. Let's test the real edge case:
        # "Go" has no 4-letter words, so words=[] → trivially satisfied
        assert requirement_satisfied("Go", "Any output here.")

    def test_stop_words_are_ignored(self):
        """Stop words (must, include, that, this, with) are not required."""
        assert requirement_satisfied(
            "Must include that this works with data",
            "The data works correctly.",
        )
