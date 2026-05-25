"""Data processing tools: CSV, JSON, DataFrames."""

import csv
import io
import json
from typing import Any

from stem_agent.tools.catalog import CatalogTool


def _csv_query(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run SQL-like operations on CSV data. Supports SELECT, WHERE, COUNT, AVG."""
    import re
    csv_text = input_data.get("csv", "")
    query = input_data.get("query", "").strip().upper()

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return {"result": [], "count": 0}

    columns = rows[0].keys()

    if query.startswith("SELECT"):
        # Simple SELECT col1, col2 FROM ...
        cols_part = query.replace("SELECT ", "").strip()
        if " FROM" in cols_part:
            cols_part = cols_part.split(" FROM")[0].strip()

        if cols_part == "*":
            selected_cols = list(columns)
        else:
            selected_cols = [c.strip() for c in cols_part.split(",")]

        # Filter with WHERE if present
        if "WHERE" in query:
            where_part = query.split("WHERE", 1)[1].strip()
            m = re.match(r"(\w+)\s*(=|>|<|>=|<=|!=)\s*(.+)", where_part)
            if m:
                col, op, val = m.group(1), m.group(2), m.group(3).strip().strip("'\"")
                rows = [r for r in rows if _compare(r.get(col, ""), op, val)]

        result = [{c: r.get(c, "") for c in selected_cols if c in r} for r in rows]
        return {"result": result, "count": len(result), "columns": selected_cols}

    if query.startswith("COUNT"):
        return {"count": len(rows)}

    return {"result": [], "error": f"Unsupported query: {query}"}


def _compare(a: str, op: str, b: str) -> bool:
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        fa, fb = a, b
    if op == "=":
        return fa == fb
    if op == ">":
        return fa > fb
    if op == "<":
        return fa < fb
    if op == ">=":
        return fa >= fb
    if op == "<=":
        return fa <= fb
    if op == "!=":
        return fa != fb
    return False


def _json_extract(input_data: dict[str, Any]) -> dict[str, Any]:
    """Extract values from nested JSON using dot-notation paths."""
    data = input_data.get("data", {})
    if isinstance(data, str):
        data = json.loads(data)
    paths = input_data.get("paths", [])
    if isinstance(paths, str):
        paths = [paths]

    result = {}
    for path in paths:
        parts = path.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                value = value[int(part)] if int(part) < len(value) else None
            else:
                value = None
                break
        result[path] = value
    return {"extracted": result}


def _dataframe_stats(input_data: dict[str, Any]) -> dict[str, Any]:
    """Compute basic statistics from a list of numeric dicts."""
    records = input_data.get("records", [])
    if not records:
        return {"count": 0, "stats": {}}

    numeric_keys = [k for k in records[0].keys() if all(
        isinstance(r.get(k), (int, float)) for r in records
    )]

    stats = {}
    for key in numeric_keys:
        values = [r[key] for r in records]
        stats[key] = {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values) if values else 0,
        }
    return {"count": len(records), "keys": list(records[0].keys()), "stats": stats}


def register(catalog):
    catalog.register(CatalogTool(
        name="csv_query",
        description="Run SELECT/COUNT queries on CSV data. Supports WHERE filters with =, >, <.",
        category="data",
        parameter_schema={
            "type": "object",
            "properties": {
                "csv": {"type": "string", "description": "CSV text content"},
                "query": {"type": "string", "description": "SQL-like query: SELECT col1,col2 or COUNT"},
            },
            "required": ["csv", "query"],
        },
        examples=[
            {"input": 'csv="a,b\\n1,2\\n3,4" query="SELECT a"', "output": "{count: 2, result: [{a:1},{a:3}]}"},
            {"input": 'csv="name,score\\nA,10\\nB,20" query="COUNT"', "output": "{count: 2}"},
        ],
        cost_estimate=0.0,
        fn=_csv_query,
    ))
    catalog.register(CatalogTool(
        name="json_extract",
        description="Extract values from nested JSON using dot-notation paths (e.g. 'user.address.city').",
        category="data",
        parameter_schema={
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "JSON object or string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Dot-notation paths to extract"},
            },
            "required": ["data", "paths"],
        },
        examples=[
            {"input": 'data={"user":{"name":"Alice"}} paths=["user.name"]', "output": "{user.name: Alice}"},
        ],
        cost_estimate=0.0,
        fn=_json_extract,
    ))
    catalog.register(CatalogTool(
        name="dataframe_stats",
        description="Compute min/max/avg/sum for numeric columns in a list of records.",
        category="data",
        parameter_schema={
            "type": "object",
            "properties": {
                "records": {"type": "array", "description": "List of dicts with numeric values"},
            },
            "required": ["records"],
        },
        examples=[
            {"input": 'records=[{"a":1},{"a":2},{"a":3}]', "output": "{count:3, stats:{a:{min:1,max:3,avg:2}}}"},
        ],
        cost_estimate=0.0,
        fn=_dataframe_stats,
    ))
