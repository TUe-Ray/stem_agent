"""Context Refiner — LLM-based purification of agent outputs between workflow steps.

Instead of passing verbose markdown essays from one agent to the next,
ContextRefiner extracts only the essential structured information,
dramatically improving signal-to-noise ratio for downstream agents.
"""

from __future__ import annotations

import json
from typing import Any

EXTRACTION_PROMPTS: dict[str, str] = {
    # ── locator output → patcher input ──
    "locate_to_patch": """\
Extract the essential debugging information from the locator's report below.
Return a JSON object with ONLY these fields (use null for any you cannot find):

{{
  "source_file": "exact file path of the buggy source code",
  "function": "function or class name to modify",
  "line_number": "approximate line number (integer or null)",
  "current_behavior": "one sentence: what the code currently does wrong",
  "fix_direction": "one sentence: what change is needed to fix it",
  "evidence_read_file": "did the locator actually read the source file? true/false/null",
  "evidence_snippet": "exact code snippet the locator reported, if any, or null"
}}

RULES:
- Strip ALL markdown boilerplate, headers, self-dialogue ("I will read...", "### Summary")
- If the locator wrote "# Some code..." or "# Existing..." as a placeholder, set evidence_read_file to false
- If line numbers are estimates ("around line 100", "likely around"), set line_number to null
- Output ONLY the JSON object, no explanation, no markdown fences

LOCATOR REPORT:
{raw_output}""",

    # ── patcher output → verifier input ──
    "patch_to_verify": """\
Extract the patch information from the patcher's output below.
Return a JSON object with ONLY these fields:

{{
  "patch": "the raw unified diff (pass through exactly as-is)",
  "has_placeholder": "true if patch contains '# Some code...', '# Existing...', or similar fake comments, else false",
  "hunk_count": "number of @@ hunk headers in the patch (integer)",
  "format_valid": "true if patch starts with '--- a/' or 'diff --git', else false"
}}

RULES:
- Pass the patch text through EXACTLY — do not modify, truncate, or reformat
- Check for placeholder comments like "# ... some code ..." or "# Some parsing logic"
- Check for malformed diff headers ("diff diff --git" → format_valid=false)
- Output ONLY the JSON object, no explanation, no markdown fences

PATCHER OUTPUT:
{raw_output}""",
}


class ContextRefiner:
    """Purifies verbose agent output into structured key-value for the next agent."""

    def __init__(self, model_client=None, enabled: bool = True):
        self._model_client = model_client  # set later via configure()
        self.enabled = enabled

    def configure(self, model_client) -> None:
        self._model_client = model_client

    def refine(
        self,
        step_id: str,
        raw_output: str,
    ) -> dict[str, Any]:
        """Refine raw agent output into structured fields for the next step.

        Args:
            step_id: The workflow step that just completed (e.g. 'locate', 'self_verifying_patch')
            raw_output: The agent's raw output text

        Returns:
            A dict with extracted fields + the original raw_output under '_raw' key.
            On failure, returns {'_raw': raw_output, '_error': '...'}
        """
        if not self.enabled or self._model_client is None:
            return {"_raw": raw_output}

        prompt_key = _step_to_prompt_key(step_id)
        template = EXTRACTION_PROMPTS.get(prompt_key)
        if template is None:
            # Unknown step — pass through as-is
            return {"_raw": raw_output}

        prompt = template.format(raw_output=raw_output)

        try:
            result = self._model_client.call(
                prompt,
                temperature=0.0,
                role="context_refiner",
            )
            if isinstance(result, dict):
                result["_raw"] = raw_output
                return result
            # Try to parse JSON from string response
            parsed = _safe_json_parse(str(result))
            parsed["_raw"] = raw_output
            return parsed
        except Exception as exc:
            return {"_raw": raw_output, "_error": str(exc)}


def _step_to_prompt_key(step_id: str) -> str:
    """Map workflow step ID to extraction prompt key."""
    mapping = {
        "locate": "locate_to_patch",
        "self_verifying_patch": "patch_to_verify",
    }
    return mapping.get(step_id, "")


def _safe_json_parse(text: str) -> dict[str, Any]:
    """Try to extract a JSON object from LLM text output."""
    text = text.strip()
    # Remove markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)
