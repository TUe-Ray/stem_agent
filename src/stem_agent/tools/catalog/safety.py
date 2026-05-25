"""Safety tools: secret detection, input validation, sandbox execution."""

import re
from typing import Any

from stem_agent.tools.catalog import CatalogTool


SECRET_PATTERNS = [
    (r'sk-[a-zA-Z0-9\-_]{20,}', "OpenAI API key"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token"),
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
    (r'AIza[0-9A-Za-z\-_]{35}', "Google API key"),
    (r'xox[bpras]-[0-9A-Za-z\-]+', "Slack token"),
    (r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----', "Private key"),
]


def _check_secrets(input_data: dict[str, Any]) -> dict[str, Any]:
    """Scan text/code for accidentally exposed secrets (API keys, tokens)."""
    text = input_data.get("text", "")
    findings = []
    for pattern, label in SECRET_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            findings.append({"type": label, "count": len(matches), "redacted": re.sub(pattern, "***REDACTED***", matches[0]) if matches else ""})

    return {
        "clean": len(findings) == 0,
        "findings": findings,
        "scanned_chars": len(text),
    }


def _validate_input(input_data: dict[str, Any]) -> dict[str, Any]:
    """Validate input against a simple schema: check required fields, types, constraints."""
    data = input_data.get("data", {})
    required = input_data.get("required_fields", [])
    constraints = input_data.get("constraints", {})

    errors = []
    for field in required:
        if field not in data or data[field] is None or data[field] == "":
            errors.append(f"Missing required field: {field}")

    for field, rule in constraints.items():
        if field not in data:
            continue
        value = data[field]
        if "min_length" in rule and isinstance(value, str) and len(value) < rule["min_length"]:
            errors.append(f"{field}: min length {rule['min_length']}, got {len(value)}")
        if "max_length" in rule and isinstance(value, str) and len(value) > rule["max_length"]:
            errors.append(f"{field}: max length {rule['max_length']}, got {len(value)}")
        if "pattern" in rule and isinstance(value, str) and not re.match(rule["pattern"], value):
            errors.append(f"{field}: pattern mismatch")

    return {"valid": len(errors) == 0, "errors": errors, "validated_fields": list(data.keys())}


def _sandbox_exec(input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute Python code in a restricted namespace. Returns stdout capture."""
    code = input_data.get("code", "")
    # Ultra-restricted builtins
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max,
        "sum": sum, "abs": abs, "round": round, "sorted": sorted,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "int": int, "float": float, "str": str, "bool": bool, "list": list,
        "dict": dict, "tuple": tuple, "set": set, "type": type,
        "print": print, "isinstance": isinstance,
    }
    import io, sys
    stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout
    namespace = {}
    try:
        exec(code, {"__builtins__": safe_builtins}, namespace)
        output = stdout.getvalue()
        return {"output": output, "vars": {k: str(v)[:200] for k, v in namespace.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": str(e), "output": stdout.getvalue()}
    finally:
        sys.stdout = old_stdout


def register(catalog):
    catalog.register(CatalogTool(
        name="check_secrets",
        description="Scan text for accidentally exposed secrets (API keys, tokens, private keys).",
        category="safety",
        parameter_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Code or text to scan"},
            },
            "required": ["text"],
        },
        examples=[
            {"input": 'text="sk-abc123..."', "output": "{clean: false, findings: [{type: 'OpenAI API key'}]}"},
        ],
        cost_estimate=0.0,
        fn=_check_secrets,
    ))
    catalog.register(CatalogTool(
        name="validate_input",
        description="Validate data against required fields + constraints (min/max length, regex pattern).",
        category="safety",
        parameter_schema={
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Data to validate"},
                "required_fields": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "object", "description": "Field→{min_length, max_length, pattern} rules"},
            },
            "required": ["data"],
        },
        examples=[
            {"input": 'data={"name":"A"} required_fields=["name","email"]', "output": "{valid: false, errors: ['Missing: email']}"},
        ],
        cost_estimate=0.0,
        fn=_validate_input,
    ))
    catalog.register(CatalogTool(
        name="sandbox_exec",
        description="Execute Python code in restricted namespace (no I/O, no network, no imports).",
        category="safety",
        parameter_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
        examples=[
            {"input": 'code="print(sum([1,2,3]))"', "output": "{output: '6'}"},
        ],
        cost_estimate=0.1,
        fn=_sandbox_exec,
    ))
