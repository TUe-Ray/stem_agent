"""SWE-bench Lite dataset loader for stem_agent.

Provides a small, fixed set of demo instances for the code-patch-generation
scenario. Full SWE-bench_Lite download via HuggingFace datasets is gated
behind explicit opt-in (n_demo_only=3 creates a smoke-testable split).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEMO_INSTANCES: list[dict[str, Any]] = [
    {
        "instance_id": "demo__simple_validation_fix",
        "repo": "demo/simple",
        "base_commit": "abc123",
        "problem_statement": (
            "The login view raises ValueError when given an empty username. "
            "Fix the login view to return a proper 400 response instead."
        ),
        "test_patch": (
            "@@ -10,3 +10,8 @@ def test_login_valid():\n"
            "+def test_login_empty_username():\n"
            "+    response = client.post('/login', {'username': '', 'password': 'test'})\n"
            "+    assert response.status_code == 400\n"
        ),
        "hints_text": "Look at the login view in auth/views.py. The issue is around line 45.",
    },
    {
        "instance_id": "demo__token_entropy_fix",
        "repo": "demo/simple",
        "base_commit": "abc123",
        "problem_statement": (
            "The password reset token generator uses a weak random source. "
            "Replace it with secrets.token_urlsafe()."
        ),
        "test_patch": (
            "@@ -0,0 +1,8 @@\n"
            "+import re\n"
            "+def test_token_entropy():\n"
            "+    token = gen.make_token(user)\n"
            "+    assert len(token) >= 43\n"
        ),
        "hints_text": "Look at tokens.py in contrib/auth/.",
    },
    {
        "instance_id": "demo__null_check_missing",
        "repo": "demo/simple",
        "base_commit": "abc123",
        "problem_statement": (
            "The profile view crashes with AttributeError when request.user is None. "
            "Add a null check and return 401 Unauthorized."
        ),
        "test_patch": (
            "@@ -5,3 +5,8 @@ def test_profile_authenticated():\n"
            "+def test_profile_anonymous():\n"
            "+    response = client.get('/profile')\n"
            "+    assert response.status_code == 401\n"
        ),
        "hints_text": "Look at profile view in users/views.py. request.user can be AnonymousUser.",
    },
]


def download_swebench_lite(
    n_train: int = 2,
    n_val: int = 1,
    n_hidden: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return train/val/hidden splits from the built-in demo instances.

    For production use, install ``datasets`` and load
    ``princeton-nlp/SWE-bench_Lite`` from HuggingFace.
    """
    instances = list(DEMO_INSTANCES)
    hashed = sorted(
        instances,
        key=lambda item: hashlib.sha256(
            (item["instance_id"] + str(seed)).encode()
        ).hexdigest(),
    )
    train = hashed[: min(n_train, len(hashed))]
    remaining = hashed[n_train:]
    val = remaining[: min(n_val, len(remaining))]
    hidden = remaining[n_val : n_val + n_hidden] if n_hidden else []
    return train, val, hidden


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
