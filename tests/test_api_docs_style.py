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


def test_checker_flags_free_form_and_replacement(tmp_path):
    """'free-form' / 'free-text' and the filler 'Replacement <thing>' prefix are
    banned; the verb 'Replaces …' (real replace-vs-merge semantics) is allowed.
    """
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    payload: dict = Field(description='Replacement free-form payload for the item')\n"
        "    other: dict = Field(description='Free-text notes')\n"
        "@router.put('/things', summary='Update thing')\n"
        "async def update_thing():\n"
        "    '''Update a thing'''\n"
        "    return []\n"
    )
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    config: dict = Field(description='New config. Replaces the stored config. Omit to leave unchanged')\n"
        "@router.put('/x', summary='Update x')\n"
        "async def update_x():\n"
        "    '''Update x'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "bad.py" in joined
    assert "free-form" in joined
    assert "Replacement" in joined
    # The verb 'Replaces the stored config' is a legitimate replace-vs-merge note.
    assert not any(
        "ok.py" in v for v in checker.find_violations(tmp_path)
    )


def test_checker_flags_ui_save_verb_and_http_plumbing_jargon(tmp_path):
    """The UI verb 'save/saving/saved' and HTTP-plumbing jargon ('hop-by-hop')
    are banned; 'stored'/'create' and plain header wording are fine.
    """
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    url: str = Field(description='Endpoint. Required when verifying before saving an agent')\n"
        "    headers: dict = Field(description='Headers (hop-by-hop headers are stripped)')\n"
        "@router.post('/things', summary='Verify agent connection before saving')\n"
        "async def verify_thing():\n"
        "    '''Verify a connection before saving an agent'''\n"
        "    return []\n"
    )
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    url: str = Field(description='Endpoint. Read from the stored config for an existing agent')\n"
        "@router.post('/x', summary='Verify an agent connection')\n"
        "async def verify_x():\n"
        "    '''Verify a connection without creating an agent'''\n"
        "    return []\n"
    )
    violations = checker.find_violations(tmp_path)
    joined = "\n".join(violations)
    assert "bad.py" in joined
    assert "save" in joined  # the UI-verb message
    assert "hop-by-hop" in joined
    assert not any("ok.py" in v for v in violations)


def test_checker_flags_trailing_period_in_description(tmp_path):
    """description= / summary= are terse labels and must not end in a period;
    docstrings (prose) keep theirs, and `...` ellipses are allowed.
    """
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter, Path\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    name: str = Field(description='Name of the thing.')\n"
        "@router.get('/x/{tid}', summary='Get the thing.')\n"
        "async def get_x(tid: str = Path(description='Thing to get.')):\n"
        "    '''Get a thing. This docstring sentence keeps its period.'''\n"
        "    return []\n"
    )
    # An f-string description and a concatenated one, both ending in a period,
    # must also be caught (regression: only bare literals were checked before).
    (tmp_path / "fstr.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "N = 500\n"
        "PART = 'Items to create'\n"
        "class Thing(BaseModel):\n"
        "    a: str = Field(description=f'Create at most {N} items.')\n"
        "    b: str = Field(description=PART + ', at most 500.')\n"
        "@router.get('/x', summary='Get thing')\n"
        "async def get_x():\n"
        "    '''Get a thing'''\n"
        "    return []\n"
    )
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    a: str = Field(description='Name of the thing')\n"
        "    b: str = Field(description='First sentence. Second with no end period')\n"
        "    c: str = Field(description='Ends in an ellipsis ...')\n"
        "    d: str = Field(description=f'At most {500} items, no end period')\n"
        "@router.get('/x', summary='Get thing')\n"
        "async def get_x():\n"
        "    '''Get a thing. Prose keeps its period.'''\n"
        "    return []\n"
    )
    violations = checker.find_violations(tmp_path)
    joined = "\n".join(violations)
    # bad.py: Field description + summary + Path description = 3; fstr.py: 2.
    assert joined.count("must not end with a period") == 5
    # The docstring period is NOT flagged, and ok.py (incl. its f-string) is clean.
    assert not any("ok.py" in v for v in violations)


def test_checker_flags_type_caveat_parenthetical(tmp_path):
    """The '(type, required/optional)' parenthetical (the '(detail)' template) is
    banned; a bare unit/format parenthetical like '(ISO 8601 UTC)' is fine.
    """
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    history: list = Field(description='`history` (array, required): the turns')\n"
        "    settings: dict = Field(description='`settings` (object, optional): extras')\n"
        "@router.post('/things', summary='Create thing')\n"
        "async def make_thing():\n"
        "    '''Create a thing'''\n"
        "    return []\n"
    )
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    created_at: str = Field(description='When it was created (ISO 8601 UTC)')\n"
        "@router.get('/x', summary='Get thing')\n"
        "async def get_thing():\n"
        "    '''Get a thing'''\n"
        "    return []\n"
    )
    violations = checker.find_violations(tmp_path)
    joined = "\n".join(violations)
    assert "bad.py" in joined and "detail" in joined
    assert not any("ok.py" in v for v in violations)


def test_checker_flags_hyphenated_adverb_participle(tmp_path):
    """Doc text must not hyphenate an adverb onto a participle ('already-linked
    tests'); rephrase as a relative clause. Domain terms ('soft-deleted') are fine.
    """
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    ids: list = Field(description='The currently-linked evaluators')\n"
        "@router.post('/things', summary='Link things')\n"
        "async def link_things():\n"
        "    '''Link things. Already-linked ones are skipped'''\n"
        "    return []\n"
    )
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/x', summary='List things')\n"
        "async def list_x():\n"
        "    '''List things, skipping soft-deleted rows'''\n"
        "    return []\n"
    )
    violations = checker.find_violations(tmp_path)
    joined = "\n".join(violations)
    # Both the Field description and the docstring compound are flagged.
    assert "bad.py" in joined and "already-linked" in joined
    # The domain term "soft-deleted" is not caught by the adverb rule.
    assert not any("ok.py" in v and "adverb" in v for v in violations)


def test_checker_flags_storage_jargon(tmp_path):
    """Doc text must describe a value's meaning, not its DB storage ('link row
    ID', 'pinned on the pivot'). Bare 'rows' for user data stays allowed.
    """
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    id: int = Field(description='Auto-increment link row ID')\n"
        "    ver: str = Field(description='Version ID pinned on the pivot at link time')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things'''\n"
        "    return []\n"
    )
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Cap(BaseModel):\n"
        "    n: int = Field(description='Max rows per eval')\n"
        "@router.get('/cap', summary='Get cap')\n"
        "async def get_cap():\n"
        "    '''Get the cap'''\n"
        "    return {}\n"
    )
    violations = checker.find_violations(tmp_path)
    joined = "\n".join(violations)
    assert "bad.py" in joined and "storage jargon" in joined
    # "Max rows per eval" is legitimate user data, not storage jargon.
    assert not any("ok.py" in v for v in violations)


def test_checker_flags_null_caveats_but_keeps_input_effects(tmp_path):
    """Nullability caveats are noise (the `| null` type shows it). But null as an
    input effect ("clears the cell") carries meaning and must survive.
    """
    (tmp_path / "bad.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    a: int = Field(None, description='Total test cases. Null until done')\n"
        "    b: str = Field(None, description='Agent config, or null if unset')\n"
        "    c: dict = Field(None, description='Aggregated cost. `null` when not reported')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things'''\n"
        "    return []\n"
    )
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Ann(BaseModel):\n"
        "    v: dict = Field(None, description='Annotation value. `null` clears the cell')\n"
        "    e: str = Field(None, description='Evaluator ID. `null` marks a row-level overall annotation')\n"
        "@router.get('/ann', summary='Get annotation')\n"
        "async def get_ann():\n"
        "    '''Get the annotation'''\n"
        "    return {}\n"
    )
    violations = checker.find_violations(tmp_path)
    joined = "\n".join(violations)
    assert "bad.py" in joined and "nullability caveat" in joined
    # Input-effect null phrasings must not be flagged.
    assert not any("ok.py" in v and "nullability" in v for v in violations)


def test_checker_flags_non_empty_validation_caveat(tmp_path):
    """'non-empty' in field docs is a validation caveat the API already enforces."""
    (tmp_path / "d.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    ids: list = Field(description='Model names to benchmark. Must be non-empty')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "non-empty" in joined


def test_checker_flags_per_x(tmp_path):
    """'per-X' compound modifiers must be rewritten as 'for each X'."""
    (tmp_path / "p.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    r: list = Field(None, description='Per-model results')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "for each X" in joined


def test_checker_allows_code_identifiers_in_doc_text(tmp_path):
    """Code refs that merely contain 'org' must not trip the terminology gate."""
    (tmp_path / "ok.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/x', summary='Get x')\n"
        "async def get_x():\n"
        "    '''Resolved via `get_current_org` (the `X-Org-UUID` header). See `/org-limits`.'''\n"
        "    return {}\n"
    )
    assert checker.find_violations(tmp_path) == []


def test_checker_flags_uuid_and_caller_in_doc_text(tmp_path):
    """Doc text must say 'ID', not 'UUID'; address reader as 'you'/'your workspace'."""
    (tmp_path / "uuid.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    name: str = Field(description='Agent UUID for the caller')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things in the caller\\'s workspace.'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "not 'UUID'" in joined
    assert "your workspace" in joined


def test_checker_flags_em_dash_and_semicolon(tmp_path):
    """Rendered doc text must not contain em-dashes or clause-splitting semicolons."""
    (tmp_path / "mech.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    a: str = Field(description='A value — explained')\n"
        "    b: str = Field(description='A value; explained further')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things.'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "no em-dashes" in joined
    assert "no clause-splitting semicolons" in joined


def test_checker_flags_unit_suffix_repeat(tmp_path):
    """A `*_ms` field must not repeat the bare 'ms' abbreviation in its description."""
    (tmp_path / "unit.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    latency_ms: float = Field(description='Response latency in ms')\n"
        "    ok_ms: float = Field(description='Response latency in milliseconds')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things.'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "latency_ms" in joined and "milliseconds" in joined
    # The spelled-out unit is fine — no violation for ok_ms.
    assert "ok_ms" not in joined


def test_checker_resolves_composed_descriptions(tmp_path):
    """A description built from a module-level constant (`_SHARED + "..."`) is
    still scanned — bans must reach it, not just plain string literals."""
    (tmp_path / "composed.py").write_text(
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "_SHARED = 'Base shape.'\n"
        "class Thing(BaseModel):\n"
        "    a: str = Field(description=_SHARED + ' Uses deep-merge on update')\n"
        "    b: str = Field(description=_SHARED + ' clause; another clause')\n"
        "@router.get('/things', summary='List things')\n"
        "async def list_things():\n"
        "    '''List things.'''\n"
        "    return []\n"
    )
    joined = "\n".join(checker.find_violations(tmp_path))
    assert "deep-merge" in joined  # jargon ban reached the composed description
    assert "clause-splitting semicolons" in joined  # mechanics ban reached it too


def test_checker_accepts_a_good_module(tmp_path):
    (tmp_path / "good.py").write_text(
        "from fastapi import APIRouter, Path\n"
        "from pydantic import BaseModel, Field\n"
        "router = APIRouter()\n"
        "class Thing(BaseModel):\n"
        "    name: str = Field(description='The name')\n"
        "@router.get('/things/{thing_id}', summary='Get thing')\n"
        "async def get_thing(thing_id: str = Path(description='The thing to retrieve')):\n"
        "    '''Retrieve a thing by id.'''\n"
        "    return {}\n"
    )
    assert checker.find_violations(tmp_path) == []
