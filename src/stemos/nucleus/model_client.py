from __future__ import annotations

import json
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised through runtime error path.
    OpenAI = None  # type: ignore[assignment]


RESPONSES_SCOPE_HINTS = (
    "api.responses.write",
    "insufficient permissions",
    "missing scopes",
)


class ModelClient:
    """Small model abstraction used by both offline and OpenAI-backed paths."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        offline: bool = True,
        endpoint: str | None = None,
    ):
        self.model = model
        self.offline = offline
        self.endpoint = endpoint or os.getenv("STEMOS_OPENAI_ENDPOINT", "auto")
        self.model_calls = 0
        self.fallback_used = False
        self.structured_output_repairs = 0

    def call(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> str | dict[str, Any]:
        if self.offline or not os.getenv("OPENAI_API_KEY"):
            return self._offline_response(prompt, response_schema, context or {})
        self.model_calls += 1
        return self._openai_response(prompt, response_schema, context or {}, tools or [])

    def record_structured_output_repair(self) -> None:
        self.structured_output_repairs += 1

    def _offline_response(
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
    ) -> str | dict[str, Any]:
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI mode requires installing stemos[openai]."
            )

        client = OpenAI()
        if self.endpoint == "chat_completions":
            return self._chat_completions_response(client, prompt, response_schema)
        if self.endpoint not in {"auto", "responses"}:
            raise RuntimeError(
                "STEMOS_OPENAI_ENDPOINT must be one of: auto, responses, chat_completions"
            )

        try:
            return self._responses_response(client, prompt, response_schema, tools)
        except Exception as exc:
            if self.endpoint == "responses" or not self._is_responses_scope_error(exc):
                raise
            self.fallback_used = True
            return self._chat_completions_response(client, prompt, response_schema)

    def _responses_response(
        self,
        client: Any,
        prompt: str,
        response_schema: dict[str, Any] | None,
        tools: list[Any],
    ) -> str | dict[str, Any]:
        if response_schema:
            response = client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "stemos_structured_output",
                        "schema": response_schema,
                        "strict": True,
                    }
                },
            )
            return json.loads(response.output_text)

        response = client.responses.create(model=self.model, input=prompt, tools=tools)
        return response.output_text

    def _chat_completions_response(
        self,
        client: Any,
        prompt: str,
        response_schema: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_schema:
            if self._has_freeform_object(response_schema):
                kwargs["messages"] = [
                    {
                        "role": "user",
                        "content": "Return only valid JSON matching the requested schema.\n"
                        + prompt,
                    }
                ]
                kwargs["response_format"] = {"type": "json_object"}
            else:
                normalized_schema = self._normalize_strict_json_schema(response_schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "stemos_structured_output",
                        "schema": normalized_schema,
                        "strict": True,
                    },
                }
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if response_schema:
            return json.loads(content)
        return content

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
