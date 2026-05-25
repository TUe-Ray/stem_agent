"""Tests for tool catalog: registration, rendering, basic tool functionality."""
import pytest

from stem_agent.tools.catalog import CatalogTool, ToolCatalog, get_catalog


class TestToolCatalog:
    def test_register_and_get(self):
        catalog = ToolCatalog()
        tool = CatalogTool(
            name="test_tool",
            description="A test tool",
            category="test",
            parameter_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            examples=[],
        )
        catalog.register(tool)
        assert catalog.get("test_tool") is tool
        assert catalog.get("nonexistent") is None

    def test_list_by_category(self):
        catalog = ToolCatalog()
        catalog.register(CatalogTool(name="t1", description="d", category="data",
                                      parameter_schema={}, examples=[]))
        catalog.register(CatalogTool(name="t2", description="d", category="data",
                                      parameter_schema={}, examples=[]))
        catalog.register(CatalogTool(name="t3", description="d", category="code",
                                      parameter_schema={}, examples=[]))
        assert len(catalog.list_by_category("data")) == 2
        assert len(catalog.list_by_category("code")) == 1
        assert len(catalog.list_by_category("nonexistent")) == 0

    def test_categories(self):
        catalog = ToolCatalog()
        catalog.register(CatalogTool(name="a", description="d", category="data",
                                      parameter_schema={}, examples=[]))
        catalog.register(CatalogTool(name="b", description="d", category="code",
                                      parameter_schema={}, examples=[]))
        assert set(catalog.categories()) == {"code", "data"}

    def test_render_for_nucleus(self):
        catalog = ToolCatalog()
        catalog.register(CatalogTool(
            name="csv_query", description="Run SQL on CSV",
            category="data",
            parameter_schema={"type": "object", "properties": {
                "csv": {"type": "string"}, "query": {"type": "string"},
            }, "required": ["csv", "query"]},
            examples=[{"input": 'csv="a" query="COUNT"', "output": "{count: 1}"}],
        ))
        rendered = catalog.render_for_nucleus()
        assert "csv_query" in rendered
        assert "Run SQL on CSV" in rendered
        assert "csv: string" in rendered

    def test_singleton_catalog_has_tools(self):
        catalog = get_catalog()
        tools = catalog.list_all()
        assert len(tools) > 0, "Singleton catalog should have registered tools"
        categories = catalog.categories()
        assert "data" in categories
        assert "code" in categories
        assert "text" in categories
        assert "files" in categories
        assert "safety" in categories
        assert "reasoning" in categories


class TestDataTools:
    def test_csv_query_select(self):
        from stem_agent.tools.catalog.data import _csv_query
        result = _csv_query({"csv": "name,score\nA,10\nB,20\nC,30", "query": "SELECT name,score"})
        assert result["count"] == 3
        assert len(result["result"]) == 3

    def test_csv_query_count(self):
        from stem_agent.tools.catalog.data import _csv_query
        result = _csv_query({"csv": "a,b\n1,2\n3,4", "query": "COUNT"})
        assert result["count"] == 2

    def test_json_extract(self):
        from stem_agent.tools.catalog.data import _json_extract
        result = _json_extract({"data": {"user": {"name": "Alice", "age": 30}}, "paths": ["user.name"]})
        assert result["extracted"]["user.name"] == "Alice"

    def test_dataframe_stats(self):
        from stem_agent.tools.catalog.data import _dataframe_stats
        result = _dataframe_stats({"records": [{"a": 1}, {"a": 2}, {"a": 3}]})
        assert result["count"] == 3
        assert result["stats"]["a"]["avg"] == 2.0


class TestTextTools:
    def test_extract_entities(self):
        from stem_agent.tools.catalog.text import _extract_entities
        result = _extract_entities({"text": "Contact alice@co.com. Budget $5,000. See https://example.com. Due 2026-06-01."})
        assert "alice@co.com" in result["emails"]
        assert "$5,000" in result["money"]
        assert "https://example.com" in result["urls"]

    def test_diff_texts(self):
        from stem_agent.tools.catalog.text import _diff_texts
        result = _diff_texts({"text_a": "hello world", "text_b": "hello new world"})
        assert result["words_added"] >= 1
        assert result["similarity"] < 1.0


class TestSafetyTools:
    def test_check_secrets_finds_key(self):
        from stem_agent.tools.catalog.safety import _check_secrets
        result = _check_secrets({"text": "OPENAI_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx"})
        assert not result["clean"]
        assert len(result["findings"]) >= 1

    def test_check_secrets_clean(self):
        from stem_agent.tools.catalog.safety import _check_secrets
        result = _check_secrets({"text": "Just normal text with no secrets."})
        assert result["clean"]

    def test_validate_input_missing_field(self):
        from stem_agent.tools.catalog.safety import _validate_input
        result = _validate_input({"data": {"name": "A"}, "required_fields": ["name", "email"]})
        assert not result["valid"]
        assert any("email" in e for e in result["errors"])


class TestReasoningTools:
    def test_verify_claim_absolute_language(self):
        from stem_agent.tools.catalog.reasoning import _verify_claim
        result = _verify_claim({"claim": "This always works perfectly."})
        assert not result["verified"]
        assert any("always" in i.lower() for i in result["issues"])
