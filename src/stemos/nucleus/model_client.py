from __future__ import annotations

import json
import os
from typing import Any


class ModelClient:
    """Small model abstraction used by both offline and OpenAI-backed paths."""

    def __init__(self, model: str = "gpt-5.5", offline: bool = True):
        self.model = model
        self.offline = offline

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
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI mode requires installing stemos[openai]."
            ) from exc

        client = OpenAI()
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
