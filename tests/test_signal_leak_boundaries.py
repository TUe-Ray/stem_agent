import pytest

from stem_agent.kernel.signal_audit import SignalLeakAuditor


def test_signal_leak_auditor_catches_raw_case_text():
    auditor = SignalLeakAuditor(forbidden_case_ids=["final_001"], forbidden_case_texts=["Create a structured plan for a postmortem after a failed launch"])
    with pytest.raises(ValueError):
        auditor.assert_no_leak_in_text("Use final_001: Create a structured plan for a postmortem after a failed launch")
