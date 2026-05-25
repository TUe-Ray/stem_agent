"""Phase C: Sandbox pathlib — reads allowed, write methods blocked (test_sandbox_pathlib)."""
import pytest

from stem_agent.kernel.sandbox import Sandbox
from stem_agent.kernel.validators import ValidationResult


@pytest.fixture
def sandbox():
    return Sandbox()


class TestPathlibAllowed:
    def test_pathlib_import_is_allowed(self, sandbox):
        """Importing pathlib does not trigger forbidden import check."""
        code = "from pathlib import Path\n\ndef run(data):\n    p = Path('/tmp/test')\n    return {'passed': p.exists()}\n"
        result = sandbox.validate_generated_tool_code(code)
        assert result.allowed, f"pathlib import should be allowed, got: {result.reason}"

    def test_path_read_methods_are_allowed(self, sandbox):
        """Read-only pathlib methods (exists, is_file, read_text,…) are not blocked."""
        code = (
            "from pathlib import Path\n"
            "def run(data):\n"
            "    p = Path('/tmp/test')\n"
            "    return {'exists': p.exists(), 'name': p.name, 'parent': str(p.parent)}\n"
        )
        result = sandbox.validate_generated_tool_code(code)
        assert result.allowed, f"read-only pathlib ops should be allowed, got: {result.reason}"


class TestPathlibWriteBlocked:
    def test_write_text_is_blocked_on_path_call(self, sandbox):
        """Path('/tmp').write_text('...') is blocked at AST level."""
        code = "from pathlib import Path\ndef run(data):\n    Path('/tmp/x').write_text('hello')\n    return {'ok': True}\n"
        result = sandbox.validate_generated_tool_code(code)
        assert not result.allowed
        assert "write_text" in result.reason

    def test_unlink_is_blocked(self, sandbox):
        """Path.unlink() is blocked."""
        code = "from pathlib import Path\ndef run(data):\n    Path('/tmp/x').unlink()\n    return {'ok': True}\n"
        result = sandbox.validate_generated_tool_code(code)
        assert not result.allowed
        assert "unlink" in result.reason

    def test_mkdir_is_blocked(self, sandbox):
        """Path.mkdir() is blocked."""
        code = "from pathlib import Path\ndef run(data):\n    Path('/tmp/dir').mkdir()\n    return {'ok': True}\n"
        result = sandbox.validate_generated_tool_code(code)
        assert not result.allowed
        assert "mkdir" in result.reason

    def test_rename_is_blocked(self, sandbox):
        """Path.rename() is blocked."""
        code = "from pathlib import Path\ndef run(data):\n    Path('/tmp/a').rename('/tmp/b')\n    return {'ok': True}\n"
        result = sandbox.validate_generated_tool_code(code)
        assert not result.allowed
        assert "rename" in result.reason
