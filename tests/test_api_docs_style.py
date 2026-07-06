"""CI gate for the `api-writing-style` skill.

Asserts every router endpoint/model follows the house style, and that the
checker actually catches violations (so a green suite means something).
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "check_api_docs_style.py"

_spec = importlib.util.spec_from_file_location("check_api_docs_style", _SCRIPT)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def test_routers_follow_api_doc_style():
    violations = checker.find_violations()
    assert violations == [], (
        "API doc-style violations found — see the api-writing-style skill:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_checker_flags_a_bad_module(tmp_path):
    """A router with a missing summary, no docstring, an undocumented path param,
    and a bare model field must produce one violation each."""
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    name: str\n"
        "@router.get('/things/{thing_id}')\n"
        "async def get_thing(thing_id: str):\n"
        "    return {}\n"
    )
    violations = checker.find_violations(tmp_path)
    joined = "\n".join(violations)
    assert "missing summary=" in joined
    assert "missing a docstring" in joined
    assert "path param 'thing_id'" in joined
    assert "Thing.name" in joined


def test_checker_flags_banned_terminology(tmp_path):
    """Doc text must say 'workspace'/'API key', never org/organization/sk_/secret."""
    (tmp_path / "term.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    name: str = Field(description='Name, unique within the org')\n"
        "    key: str = Field(description='The raw sk_ secret for the organization')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things for the caller org.'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "not 'org'" in joined
    assert "not 'organization'" in joined
    assert "sk_" in joined
    assert "not 'secret'" in joined


def test_checker_allows_code_identifiers_in_doc_text(tmp_path):
    """Code refs that merely contain 'org' must not trip the terminology gate."""
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/x', summary='Get x')\n"
        "async def get_x():\n"
        "    '''Resolved via `get_current_org` (the `X-Org-UUID` header); see `/org-limits`.'''\n"
        "    return {}\n"
    )
    assert checker.find_violations(tmp_path) == []


def test_checker_accepts_a_good_module(tmp_path):
    (tmp_path / "good.py").write_text(
        "from fastapi import APIRouter, Path\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    name: str = Field(description='The name')\n"
        "@router.get('/things/{thing_id}', summary='Get thing')\n"
        "async def get_thing(thing_id: str = Path(description='Thing UUID')):\n"
        "    '''Retrieve a thing by id.'''\n"
        "    return {}\n"
    )
    assert checker.find_violations(tmp_path) == []
