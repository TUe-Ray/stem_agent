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
        return self._openai_response(prompt, response_schema, context or {}, tools or [])

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
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "stemos_structured_output",
                    "schema": response_schema,
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
