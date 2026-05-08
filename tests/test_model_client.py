import json

from stemos.config import Settings
from stemos.nucleus import model_client as model_client_module
from stemos.nucleus.model_client import ModelClient


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeChatResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeResponses:
    def create(self, **kwargs):
        raise RuntimeError(
            "OpenAI API returned HTTP 401: Missing scopes: api.responses.write"
        )


class _FakeCompletions:
    last_kwargs = None

    def create(self, **kwargs):
        _FakeCompletions.last_kwargs = kwargs
        if "response_format" in kwargs:
            return _FakeChatResponse(json.dumps({"ok": True}))
        return _FakeChatResponse("OK")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self):
        self.responses = _FakeResponses()
        self.chat = _FakeChat()


def test_model_client_auto_falls_back_to_chat_completions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(model_client_module, "OpenAI", _FakeOpenAI, raising=False)

    client = ModelClient(model="gpt-4.1-mini", offline=False, endpoint="auto")

    assert client.call("Say OK") == "OK"


def test_model_client_chat_completions_structured_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(model_client_module, "OpenAI", _FakeOpenAI, raising=False)

    client = ModelClient(model="gpt-4.1-mini", offline=False, endpoint="chat_completions")
    result = client.call(
        "Return JSON",
        response_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )

    assert result == {"ok": True}


def test_settings_default_model_is_chat_completions_friendly(monkeypatch):
    monkeypatch.delenv("STEM_AGENT_MODEL", raising=False)

    assert Settings().model == "gpt-4.1-mini"


def test_strict_chat_schema_disallows_extra_properties_recursively():
    client = ModelClient()
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {"type": "string"},
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    }

    normalized = client._normalize_strict_json_schema(schema)

    assert normalized["additionalProperties"] is False
    assert normalized["required"] == ["outer", "items"]
    assert normalized["properties"]["outer"]["additionalProperties"] is False
    assert normalized["properties"]["outer"]["required"] == ["inner"]
    assert normalized["properties"]["items"]["items"]["additionalProperties"] is False
    assert normalized["properties"]["items"]["items"]["required"] == ["name"]


def test_chat_completions_uses_json_object_for_freeform_patch(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(model_client_module, "OpenAI", _FakeOpenAI, raising=False)

    client = ModelClient(model="gpt-4.1-mini", offline=False, endpoint="chat_completions")
    client.call(
        "Return JSON",
        response_schema={
            "type": "object",
            "properties": {"patch": {"type": "object"}},
            "required": ["patch"],
        },
    )

    assert _FakeCompletions.last_kwargs["response_format"] == {"type": "json_object"}
