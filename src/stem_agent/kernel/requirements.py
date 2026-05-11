from __future__ import annotations

import re


SECTION_NEEDLES: dict[str, list[str]] = {
    "summary": ["summary"],
    "step": ["steps", "1.", "- "],
    "final": ["final answer", "final output", "recommendation"],
    "acceptance": ["acceptance criteria", "acceptance"],
    "qa": ["qa report", "quality review", "review"],
    "decision": ["decision log", "decision"],
    "artifact": ["artifact", ".md"],
}

_STOP_WORDS = {"must", "include", "with", "that", "this", "output"}


def requirement_section_satisfied(requirement: str, output: str) -> bool | None:
    req = requirement.lower()
    out = output.lower()
    for key, needles in SECTION_NEEDLES.items():
        if key in req:
            return any(needle in out for needle in needles)
    return None


def requirement_satisfied(requirement: str, output: str) -> bool:
    section_result = requirement_section_satisfied(requirement, output)
    if section_result is not None:
        return section_result

    req = requirement.lower()
    out = output.lower()
    words = [
        word
        for word in re.findall(r"[a-zA-Z]{4,}", req)
        if word not in _STOP_WORDS
    ]
    if not words:
        return True
    return sum(1 for word in words if word in out) >= max(1, len(words) // 2)
