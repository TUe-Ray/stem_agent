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
from typing import Any

MAX_TOOL_STEPS = 10  # bounded loop — never while True (prevents pattern prison + cost runaway)


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
                # Prepend tool call summary if tools were used
                if all_tool_calls:
                    summary = self._tool_usage_summary(all_tool_calls)
                    text = summary + "\n\n" + text
                return text

            # ── Model wants to call tools ──
            assistant_msg = choice.message
            messages.append(assistant_msg)  # assistant message with tool_calls

            for tool_call in (assistant_msg.tool_calls or []):
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id

                # Parse arguments (JSON string → dict)
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Execute tool
                result = self._execute_tool(tool_name, args)

                all_tool_calls.append({
                    "name": tool_name,
                    "args": args,
                    "result_summary": str(result)[:200],
                    "success": not str(result).startswith("Error"),
                })

                # Append tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result, ensure_ascii=False)
                    if isinstance(result, dict) else str(result),
                })

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
