"""Tool execution loop — enables agents to actually call tools via OpenAI tool_choice.

Before this module, stem_agent agents were single-shot: one prompt → one text response.
The LLM could describe what tools it *would* call, but they were never executed.

This module adds the standard tool execution loop:

    for step in range(MAX_TOOL_STEPS):
        response = LLM(messages, tools)
        if response.tool_calls:
            execute each tool → append results to messages → loop
        else:
            return text response

Reference: OpenAI function calling guide, Anthropic tool use docs, Koog framework.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

MAX_TOOL_STEPS = 5  # bounded loop — more steps = more TPM usage with gpt-4o-mini
MAX_RESULT_CHARS = 4000  # per-tool-result cap — prevents giant read_file outputs from blowing context
# gpt-4o-mini has 128K context window. 200K TPM is per-minute, not per-request.
# SWE-bench locator prompts alone are ~15-20K chars (scenario + case_input + instructions).
# Set high enough to fit prompt + 4 tool results without triggering aggressive prune.
MAX_CONTEXT_CHARS = 60000


class ToolExecutor:
    """Wraps an OpenAI client to run the tool-calling loop for agent turns.

    Each agent turn becomes:
        prompt → [tool_call → tool_result → ...] → final text response

    The caller (RoleRunner) doesn't need to change — it still gets back a string.
    """

    def __init__(self, client: Any, model: str, tools: dict[str, Any]):
        self._client = client
        self._model = model
        self._tools_map = tools  # name → stem_agent Tool object

    def run(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        """Execute the tool-calling loop and return the final text response."""
        messages = self._build_initial_messages(system_prompt, user_prompt)
        tool_schemas = self._build_tool_schemas()
        all_tool_calls: list[dict[str, Any]] = []

        for step in range(MAX_TOOL_STEPS):
            # Rate-limit to stay under gpt-4o-mini 200K TPM
            if step > 0:
                time.sleep(2.0)

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tool_schemas if step == 0 else tool_schemas,
                tool_choice="auto" if step == 0 else "auto",
                temperature=temperature if temperature is not None else 0.0,
            )
            choice = response.choices[0]

            # ── Model returned text (no tool calls) → done ──
            if choice.finish_reason != "tool_calls" and not choice.message.tool_calls:
                text = choice.message.content or ""
                if all_tool_calls:
                    summary = self._tool_usage_summary(all_tool_calls)
                    text = summary + "\n\n" + text
                # Auto-extract diff if model forgot to call write_patch
                text = self._auto_save_diff(text)
                return text

            # ── Model wants to call tools ──
            assistant_msg = choice.message
            # Build a clean message dict with only the fields OpenAI needs.
            # model_dump(exclude_none=False) includes bloat (refusal, audio,
            # function_call as explicit nulls) that wastes tokens and pushes
            # context over the prune threshold faster.
            msg: dict[str, Any] = {"role": "assistant", "content": assistant_msg.content}
            if assistant_msg.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
            messages.append(msg)

            for tool_call in (assistant_msg.tool_calls or []):
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id

                # Parse arguments
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Execute tool
                started = time.time()
                result = self._execute_tool(tool_name, args)
                elapsed = (time.time() - started) * 1000
                result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)

                # ── Langfuse tracing: log tool call ──
                from stem_agent.observability.langfuse_tracer import log_tool_call as _log_tool
                _log_tool(
                    tool_name=tool_name,
                    args=args,
                    result_summary=result_str[:200],
                    success=not str(result).startswith("Error"),
                    duration_ms=elapsed,
                )

                # Truncate huge results
                if len(result_str) > MAX_RESULT_CHARS:
                    result_str = result_str[:MAX_RESULT_CHARS] + f"\n... [truncated, original: {len(result_str)} chars]"

                all_tool_calls.append({
                    "name": tool_name,
                    "args": args,
                    "result_summary": result_str[:200],
                    "success": not str(result).startswith("Error"),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                })

            # ── Prune context AFTER all tool results are appended ──
            # Doing this inside the for loop (per tool result) causes partial
            # pruning: assistant gets trimmed but remaining tool results are
            # appended as orphans → OpenAI 400 BadRequestError.
            messages = self._prune_context(messages)

        # ── Max steps reached ──
        # Ask model to summarize what it has so far
        messages.append({
            "role": "user",
            "content": "You've reached the maximum number of tool calls. "
                       "Please provide your best answer now without calling any more tools.",
        })
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature if temperature is not None else 0.0,
            )
            text = response.choices[0].message.content or ""
            if all_tool_calls:
                text = self._tool_usage_summary(all_tool_calls) + "\n\n" + text
            # Auto-extract diff if model forgot to call write_patch
            text = self._auto_save_diff(text)
            return text
        except Exception:
            return "Error: Tool execution loop exceeded maximum steps and final summary failed."

    def _build_initial_messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert stem_agent Tool objects to OpenAI Chat Completions tool schemas.

        Chat Completions format:
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        schemas = []
        for name, tool in self._tools_map.items():
            if name == "call_model":
                continue  # Internal tool — never expose to LLM
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": self._sanitize_parameters(tool.parameters),
                },
            }
            schemas.append(schema)
        return schemas

    def _sanitize_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Ensure parameters schema is valid for OpenAI tool calling."""
        cleaned = dict(params)
        if cleaned.get("type") != "object":
            cleaned["type"] = "object"
        # OpenAI requires properties dict and rejects additionalProperties: true
        if "properties" not in cleaned:
            cleaned["properties"] = {}
        cleaned["additionalProperties"] = False
        return cleaned

    def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with given arguments."""
        tool = self._tools_map.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}", "available": list(self._tools_map.keys())}
        try:
            return tool.run(args)
        except Exception as exc:
            return {"error": f"Tool '{name}' failed: {exc}"}

    def _tool_usage_summary(self, calls: list[dict[str, Any]]) -> str:
        """Generate a compact summary of tool calls made during this turn."""
        lines = [f"[Tool calls: {len(calls)}]"]
        for c in calls:
            status = "✓" if c["success"] else "✗"
            args_preview = json.dumps(c["args"], ensure_ascii=False)
            if len(args_preview) > 80:
                args_preview = args_preview[:77] + "..."
            lines.append(f"  {status} {c['name']}({args_preview})")
        return "\n".join(lines)

    @staticmethod
    def _extract_diff_from_text(text: str) -> str | None:
        """Extract a unified diff from model text output.

        gpt-4o-mini often outputs diffs as ```diff blocks or raw --- a/ lines
        instead of calling write_patch. This extracts them so they can be saved.

        Returns the diff text if found, None otherwise.
        """
        # Pattern 1: ```diff ... ``` code block
        m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Pattern 2: --- a/... lines (raw diff without markdown fences)
        m = re.search(r"(?:^|\n)(--- a/.*?\n(?:\+\+\+ b/.*?\n)?@@.*?(?:\n[ +-].*?)*)", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Pattern 3: diff --git a/... b/... format
        m = re.search(r"(diff --git a/.*?)(?:\n\n\n|\n```|\Z)", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        return None

    def _auto_save_diff(self, text: str) -> str:
        """Scan text for diff content and auto-save to predicted.patch if found."""
        diff_text = self._extract_diff_from_text(text)
        if not diff_text:
            return text

        # Try to save using write_patch tool
        wp_tool = self._tools_map.get("write_patch")
        if wp_tool:
            try:
                result = wp_tool.run({"patch_text": diff_text})
                if not str(result).startswith("Error"):
                    return text + "\n\n[⚡ Auto-saved diff to artifacts/predicted.patch]"
            except Exception:
                pass

        # Fallback: save directly if we know the workspace
        # (write_patch writes to artifacts/predicted.patch relative to CWD)
        try:
            patch_path = Path("artifacts") / "predicted.patch"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(diff_text)
            return text + "\n\n[⚡ Auto-saved diff to artifacts/predicted.patch]"
        except Exception:
            pass

        return text

    def _prune_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep messages under MAX_CONTEXT_CHARS while preserving tool call/result pairing.

        The OpenAI API requires every assistant tool_call to have a matching tool
        result message. Pruning must keep these as atomic units.
        """
        total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        if total <= MAX_CONTEXT_CHARS:
            return messages

        # Always keep system message (index 0) and first user message (index 1)
        keep = messages[:2]
        remaining = messages[2:]

        # Build paired units: (assistant_with_tool_calls, tool_results...)
        # If assistant has tool_calls, it must stay with its tool results
        units = []
        i = 0
        while i < len(remaining):
            msg = remaining[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Collect assistant + all following tool results
                unit = [msg]
                i += 1
                while i < len(remaining) and remaining[i].get("role") == "tool":
                    unit.append(remaining[i])
                    i += 1
                units.append(unit)
            else:
                units.append([msg])
                i += 1

        # Keep most recent units that fit in budget
        current_size = sum(len(json.dumps(m, ensure_ascii=False)) for m in keep)
        kept_units = []
        for unit in reversed(units):
            unit_size = sum(len(json.dumps(m, ensure_ascii=False)) for m in unit)
            if current_size + unit_size <= MAX_CONTEXT_CHARS:
                kept_units.insert(0, unit)
                current_size += unit_size
            else:
                break

        trimmed = sum(len(u) for u in units) - sum(len(u) for u in kept_units)
        if trimmed > 0:
            note = {
                "role": "system",
                "content": f"[{trimmed} earlier tool interactions were trimmed to stay within context limits]"
            }
            keep.append(note)

        for unit in kept_units:
            keep.extend(unit)

        return keep
