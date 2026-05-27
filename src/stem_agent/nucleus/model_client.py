from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from stem_agent.llm.client import LLMCallLogger

# OpenAI client class — resolved lazily so Langfuse env vars are loaded first
OpenAI = None  # Use _get_openai_client() instead of accessing directly

def _get_openai_client():
    """Return the OpenAI client class (Langfuse-wrapped if configured)."""
    global OpenAI
    if OpenAI is not None:
        return OpenAI
    try:
        from stem_agent.observability.langfuse_tracer import get_openai
        OpenAI = get_openai()
    except Exception:
        from openai import _OpenAI as Fallback
        OpenAI = Fallback
    return OpenAI


RESPONSES_SCOPE_HINTS = (
    "api.responses.write",
    "insufficient permissions",
    "missing scopes",
)


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


class ModelClient:
    """Small model abstraction for OpenAI API calls, with an internal test double.
    
    Supports DeepSeek via DEEPSEEK_API_KEY env var. When set, OPENAI_API_KEY
    and OPENAI_BASE_URL are auto-configured to use DeepSeek's API.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        test_mode: bool = False,
        endpoint: str | None = None,
        run_dir: str | Path | None = None,
    ):
        self.model = model
        self.test_mode = test_mode
        self.endpoint = endpoint or _env("STEM_AGENT_OPENAI_ENDPOINT", "auto")
        self.model_calls = 0
        self.fallback_used = False
        self.responses_api_available: bool | None = None
        self.structured_output_repairs = 0
        self.call_logger = LLMCallLogger(run_dir)
        self._auto_configure_deepseek()

    def _auto_configure_deepseek(self) -> None:
        """Auto-detect DeepSeek API key and configure OpenAI client for it."""
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not deepseek_key:
            return
        # Only auto-switch if model starts with deepseek/ or no OpenAI key is set
        if self.model.startswith("deepseek/") or not os.getenv("OPENAI_API_KEY"):
            os.environ.setdefault("OPENAI_API_KEY", deepseek_key)
            os.environ.setdefault("OPENAI_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
            if self.model.startswith("deepseek/"):
                self.model = self.model.split("/", 1)[1]  # strip "deepseek/" prefix
            elif not self.model or self.model == "gpt-4.1-mini":
                self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def configure_run(self, run_dir: str | Path | None) -> None:
        self.call_logger.configure(run_dir)

    def call(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        role: str | None = None,
    ) -> str | dict[str, Any]:
        full_prompt = self._full_prompt(system_prompt, prompt)
        if self.test_mode:
            response = self._test_response(prompt, response_schema, context or {})
            self._record_call(role, full_prompt, response)
            return response
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required. Copy .env.example to .env, set your key, "
                "then run: set -a; source .env; set +a"
            )
        self.model_calls += 1

        # ── Tool execution loop (Chat Completions mode only) ──
        if tools and self.endpoint == "chat_completions":
            response = self._call_with_tool_loop(
                prompt, tools, system_prompt=system_prompt, temperature=temperature
            )
            self._record_call(role, full_prompt, response)
            return response

        response, tokens_used = self._openai_response(
            prompt,
            response_schema,
            context or {},
            tools or [],
            system_prompt=system_prompt,
            temperature=temperature,
        )
        self._record_call(role, full_prompt, response, tokens_used=tokens_used)
        return response

    def _call_with_tool_loop(
        self,
        prompt: str,
        tools: list[Any],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Execute the tool-calling loop for agent turns."""
        from stem_agent.harness.tool_executor import ToolExecutor

        if _get_openai_client() is None:
            raise RuntimeError("OpenAI mode requires installing stem_agent[openai].")

        client = _get_openai_client()()
        tools_map = {t.name: t for t in tools}
        executor = ToolExecutor(client, self.model, tools_map)
        return executor.run(
            system_prompt=system_prompt or "",
            user_prompt=prompt,
            temperature=temperature,
        )

    def audit_call(
        self,
        *,
        role: str,
        system_prompt: str,
        prompt: str,
        response: str | dict[str, Any],
    ) -> None:
        self._record_call(role, self._full_prompt(system_prompt, prompt), response)

    def record_structured_output_repair(self) -> None:
        self.structured_output_repairs += 1

    def _test_response(
        self,
        prompt: str,
        response_schema: dict[str, Any] | None,
        context: dict[str, Any],
    ) -> str | dict[str, Any]:
        if response_schema:
            return {}
        return context.get("fallback_text") or prompt

    def _openai_response(
        self,
        prompt: str,
        response_schema: dict[str, Any] | None,
        context: dict[str, Any],
        tools: list[Any],
        *,
        system_prompt: str | None,
        temperature: float | None,
    ) -> tuple[str | dict[str, Any], int | None]:
        if _get_openai_client() is None:
            raise RuntimeError(
                "OpenAI mode requires installing stem_agent[openai]."
            )

        client = _get_openai_client()()
        if self.endpoint == "chat_completions":
            return self._chat_completions_response(
                client,
                prompt,
                response_schema,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        if self.endpoint not in {"auto", "responses"}:
            raise RuntimeError(
                "STEM_AGENT_OPENAI_ENDPOINT must be one of: auto, responses, chat_completions"
            )

        try:
            return self._responses_response(
                client,
                prompt,
                response_schema,
                tools,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        except Exception as exc:
            if self.endpoint == "responses" or not self._is_responses_scope_error(exc):
                raise
            self.fallback_used = True
            self.responses_api_available = False
            return self._chat_completions_response(
                client,
                prompt,
                response_schema,
                system_prompt=system_prompt,
                temperature=temperature,
            )

    def _responses_response(
        self,
        client: Any,
        prompt: str,
        response_schema: dict[str, Any] | None,
        tools: list[Any],
        *,
        system_prompt: str | None,
        temperature: float | None,
    ) -> tuple[str | dict[str, Any], int | None]:
        input_payload: str | list[dict[str, str]]
        if system_prompt:
            input_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            input_payload = prompt
        kwargs: dict[str, Any] = {"model": self.model, "input": input_payload}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_schema:
            response = client.responses.create(
                **kwargs,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "stem_agent_structured_output",
                        "schema": response_schema,
                        "strict": True,
                    }
                },
            )
            self.responses_api_available = True
            return json.loads(response.output_text), self._tokens_from_response(response)

        api_tools = [t.to_api_dict() if hasattr(t, "to_api_dict") else t for t in tools]
        response = client.responses.create(**kwargs, tools=api_tools)
        self.responses_api_available = True
        return response.output_text, self._tokens_from_response(response)

    def _chat_completions_response(
        self,
        client: Any,
        prompt: str,
        response_schema: dict[str, Any] | None,
        *,
        system_prompt: str | None,
        temperature: float | None,
    ) -> tuple[str | dict[str, Any], int | None]:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_schema:
            if self._has_freeform_object(response_schema):
                return self._try_json_object_chat_completion(
                    client, prompt, system_prompt, kwargs,
                    response_schema=response_schema,
                )
            try:
                return self._try_structured_chat_completion(
                    client, prompt, response_schema, system_prompt, kwargs
                )
            except Exception:
                # Provider doesn't support json_schema — fall back to json_object
                self.record_structured_output_repair()
                return self._try_json_object_chat_completion(
                    client, prompt, system_prompt, kwargs,
                    response_schema=response_schema,
                )
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if response_schema:
            return json.loads(content), self._tokens_from_response(response)
        return content, self._tokens_from_response(response)

    def _try_structured_chat_completion(
        self,
        client: Any,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        kwargs: dict[str, Any],
    ) -> tuple[str | dict[str, Any], int | None]:
        normalized_schema = self._normalize_strict_json_schema(response_schema)
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "stem_agent_structured_output",
                "schema": normalized_schema,
                "strict": True,
            },
        }
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return json.loads(content), self._tokens_from_response(response)

    def _try_json_object_chat_completion(
        self,
        client: Any,
        prompt: str,
        system_prompt: str | None,
        kwargs: dict[str, Any],
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[str | dict[str, Any], int | None]:
        schema_hint = self._format_schema_hint(response_schema) if response_schema else ""
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": "user", "content": schema_hint + "Return only valid JSON.\n" + prompt}]
        kwargs["messages"] = messages
        kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return json.loads(content), self._tokens_from_response(response)

    def _format_schema_hint(self, schema: dict[str, Any]) -> str:
        """Build a compact prompt hint for the JSON fields the schema requires.

        Used as a fallback when the provider doesn't support strict json_schema
        (e.g. DeepSeek). Gives the LLM enough context to produce valid output.
        """
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if not required and not properties:
            return ""
        lines = ["Your response must be a JSON object with these fields:"]
        for field in required:
            prop = properties.get(field, {})
            desc = prop.get("description", prop.get("title", ""))
            ftype = prop.get("type", "string")
            items_info = ""
            if ftype == "array" and "items" in prop:
                items_type = prop["items"].get("type", "string")
                items_info = f" of {items_type}"
            lines.append(f"  - \"{field}\" ({ftype}{items_info}): {desc}")
        # Also note optional fields
        optional = [f for f in properties if f not in required]
        if optional:
            lines.append("Optional fields:")
            for field in optional:
                prop = properties.get(field, {})
                desc = prop.get("description", prop.get("title", ""))
                ftype = prop.get("type", "string")
                items_info = ""
                if ftype == "array" and "items" in prop:
                    items_type = prop["items"].get("type", "string")
                    items_info = f" of {items_type}"
                lines.append(f"  - \"{field}\" ({ftype}{items_info}): {desc}")
        lines.append("Respond with ONLY the JSON object, nothing else.\n")
        return "\n".join(lines) + "\n"

    def _full_prompt(self, system_prompt: str | None, prompt: str) -> str:
        if not system_prompt:
            return prompt
        return f"{system_prompt}\n\n{prompt}"

    def _record_call(
        self,
        role: str | None,
        prompt: str,
        response: str | dict[str, Any],
        *,
        tokens_used: int | None = None,
    ) -> None:
        if role not in {"nucleus", "guardian"}:
            return
        response_text = response if isinstance(response, str) else json.dumps(response, sort_keys=True)
        self.call_logger.record(
            role=role,
            prompt=prompt,
            response=response_text,
            tokens_used=tokens_used,
        )

    def _tokens_from_response(self, response: Any) -> int | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        for name in ["total_tokens", "total_token_count"]:
            value = getattr(usage, name, None)
            if value is not None:
                return int(value)
        if isinstance(usage, dict):
            for name in ["total_tokens", "total_token_count"]:
                if usage.get(name) is not None:
                    return int(usage[name])
        return None

    def _is_responses_scope_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(hint in text for hint in RESPONSES_SCOPE_HINTS)

    def _normalize_strict_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Make Pydantic JSON schema acceptable to strict Chat Completions."""
        normalized = json.loads(json.dumps(schema))
        self._force_no_additional_properties(normalized)
        return normalized

    def _has_freeform_object(self, schema: Any) -> bool:
        if isinstance(schema, dict):
            if (
                schema.get("type") == "object"
                and not schema.get("properties")
                and schema.get("additionalProperties") is not False
            ):
                return True
            return any(self._has_freeform_object(value) for value in schema.values())
        if isinstance(schema, list):
            return any(self._has_freeform_object(item) for item in schema)
        return False

    def _force_no_additional_properties(self, node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}).keys())
            for value in node.values():
                self._force_no_additional_properties(value)
        elif isinstance(node, list):
            for item in node:
                self._force_no_additional_properties(item)
