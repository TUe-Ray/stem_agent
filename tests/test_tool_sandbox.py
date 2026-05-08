from stem_agent.kernel.sandbox import Sandbox


def test_sandbox_rejects_forbidden_imports():
    code = "import os\n\ndef run(input_data):\n    return {'cwd': os.getcwd()}\n"

    result = Sandbox().validate_generated_tool_code(code)

    assert result.allowed is False
    assert "Forbidden import: os" in result.reason


def test_sandbox_accepts_safe_generated_tool_code():
    code = (
        "from typing import Dict, Any\n\n"
        "def run(input_data: Dict[str, Any]) -> Dict[str, Any]:\n"
        "    return {'passed': True}\n"
    )

    result = Sandbox().validate_generated_tool_code(code)

    assert result.allowed is True


def test_sandbox_rejects_shell_command_without_timeout():
    result = Sandbox().validate_shell_command({"command": "python script.py"})

    assert result.allowed is False
    assert "timeout" in result.reason
