from requirement_sections_checker import run

def test_requirement_sections_checker_detects_missing_requirement():
    result = run({
        "output": "Summary: hello",
        "requirements": ["Must include a short summary", "Must include concrete steps"],
    })
    assert result["passed"] is False
    assert "Must include concrete steps" in result["missing"]

def test_requirement_sections_checker_accepts_covered_output():
    result = run({
        "output": "Summary with concrete steps and final answer",
        "requirements": ["Must include summary", "Must include concrete steps"],
    })
    assert result["score"] >= 0.5
