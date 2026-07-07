#!/usr/bin/env python3
"""Static enforcement of the `api-writing-style` skill.

Parses every router module under `src/routers/` (AST only — no imports, no app
boot) and flags routes/models that violate the house style:

  1. Every route (`@router.<method>(...)`) has a non-empty `summary=`.
  2. Every route function has a docstring.
  3. Every path parameter (`{x}` in the route path) whose name appears in the
     function signature is documented via `Path(...)`/`PathParam(...)` with a
     `description`.
  4. Every field on a Pydantic model (a `BaseModel` subclass, transitively) has a
     `Field(..., description=...)` with a non-empty description.
  5. Rendered doc text (descriptions, summaries, route/model/module docstrings)
     contains no banned terms, no em-dashes, and no clause-splitting semicolons.
  6. A field whose name ends in a unit suffix (e.g. `_ms`) doesn't repeat that
     unit abbreviation in its description.

Run directly (`uv run python scripts/check_api_docs_style.py`) — exits non-zero
and prints one line per violation. `tests/test_api_docs_style.py` reuses
`find_violations()` so the same gate runs in CI.

The skill (`.cursor/skills/api-writing-style/SKILL.md`) owns wording quality
(verb vocabulary, conciseness); this script only enforces the mechanical,
non-flaky invariants above.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Optional

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PATH_PARAM_FUNCS = {"Path", "PathParam"}
_PATH_TOKEN = re.compile(r"\{([^{}:]+)(?::[^{}]+)?\}")

# Terminology gate for user-facing doc text (see the api-writing-style skill).
# The lookarounds exempt code identifiers/paths (org_uuid, get_current_org,
# X-Org-UUID, /org-limits, ...) — only the standalone prose word is banned.
_BANNED_TERMS = [
    (re.compile(r"(?<![\w`\-])[Oo]rgs?(?![\w`\-])"), "say 'workspace', not 'org'"),
    (re.compile(r"(?<![\w`\-])[Oo]rganizations?(?![\w`\-])"), "say 'workspace', not 'organization'"),
    (re.compile(r"(?<!\w)sk_"), "don't reference the raw 'sk_…' key — say 'API key'"),
    (re.compile(r"\bsecret\b"), "say 'API key', not 'secret'"),
    (re.compile(r"(?<![\w`/])\bUUIDs?\b(?![\w`])"), "say 'ID', not 'UUID' in user-facing docs"),
    (re.compile(r"\bthe caller(?:'s)?\b"), "address the reader directly ('you' / 'your workspace')"),
    (re.compile(r"\bcaller's workspace\b"), "say 'your workspace'"),
    (re.compile(r"\b8-char\b"), "IDs are UUID v4 (36 chars) — don't say '8-char'"),
]

# Mechanical bans on rendered doc text (see "Brevity & mechanics" in the skill).
# Applied to the same doc-text set as _BANNED_TERMS (descriptions, summaries,
# route/model/module docstrings). Code comments are exempt — they don't render.
_MECHANICS = [
    (
        re.compile("—"),  # em-dash
        "no em-dashes in rendered docs: use a period, comma, colon, or parentheses",
    ),
    (
        re.compile(r";\s"),  # clause-splitting semicolon
        "no clause-splitting semicolons in docs: use a full stop",
    ),
    (
        re.compile(r"(?i)deep[- ]merge"),  # implementation jargon
        "don't say 'deep-merge' in docs (implementation jargon); describe the effect",
    ),
    (
        re.compile(r"(?i)human-readable"),  # filler
        "'Human-readable' is filler; drop it",
    ),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTERS_DIR = REPO_ROOT / "src" / "routers"


def _has_description_kw(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "description":
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return bool(value.value.strip())
            # Non-literal (f-string, variable) — accept; can't statically judge.
            return True
    return False


# Field-name suffix → the unit abbreviation that must NOT be repeated in prose
# (the field name already carries it; spell the word out or drop it). See
# "Don't repeat the unit that's already in the field name" in the skill.
_UNIT_SUFFIXES = [("_ms", re.compile(r"\bms\b"), "milliseconds")]


def _field_description_text(call: ast.Call) -> Optional[str]:
    """The literal `description=` string on a Field call, or None if absent /
    non-literal (f-string, variable — can't statically inspect)."""
    for kw in call.keywords:
        if kw.arg == "description" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _is_field_call(node: Optional[ast.expr]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else (
        func.attr if isinstance(func, ast.Attribute) else None
    )
    return name == "Field"


def _model_class_names(tree: ast.Module) -> set[str]:
    """Names of classes that are Pydantic models (inherit BaseModel, transitively)."""
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    def base_names(cls: ast.ClassDef) -> set[str]:
        names = set()
        for base in cls.bases:
            if isinstance(base, ast.Name):
                names.add(base.id)
            elif isinstance(base, ast.Attribute):
                names.add(base.attr)
        return names

    models = {c.name for c in classes if "BaseModel" in base_names(c)}
    changed = True
    while changed:
        changed = False
        for c in classes:
            if c.name not in models and base_names(c) & models:
                models.add(c.name)
                changed = True
    return models


def _check_model_fields(cls: ast.ClassDef, rel: str) -> list[str]:
    out = []
    for stmt in cls.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        field = stmt.target.id
        if field == "model_config":
            continue
        ann = stmt.annotation
        if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == "ClassVar":
            continue
        loc = f"{rel}:{stmt.lineno} {cls.name}.{field}"
        if not _is_field_call(stmt.value):
            out.append(f"{loc}: field is not documented with Field(description=...)")
        elif not _has_description_kw(stmt.value):
            out.append(f"{loc}: Field(...) is missing a non-empty description=")
        else:
            desc = _field_description_text(stmt.value)
            if desc:
                for suffix, pat, word in _UNIT_SUFFIXES:
                    if field.endswith(suffix) and pat.search(desc):
                        out.append(
                            f"{loc}: unit abbreviation repeats the '{suffix}' "
                            f"field-name suffix — say '{word}' or drop it"
                        )
    return out


def _arg_defaults(func) -> dict[str, ast.expr]:
    """Map every argument name to its default node (or None)."""
    a = func.args
    defaults: dict[str, ast.expr] = {}
    posonly = list(a.posonlyargs) + list(a.args)
    for i, arg in enumerate(posonly):
        defaults[arg.arg] = None
    # positional defaults align to the tail of posonly+args
    if a.defaults:
        tail = posonly[-len(a.defaults):]
        for arg, dflt in zip(tail, a.defaults):
            defaults[arg.arg] = dflt
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        defaults[arg.arg] = dflt
    return defaults


def _check_route(func, decorator: ast.Call, rel: str) -> list[str]:
    out = []
    name = func.name
    loc = f"{rel}:{func.lineno} {name}"

    summary = next((kw for kw in decorator.keywords if kw.arg == "summary"), None)
    if summary is None:
        out.append(f"{loc}: route decorator is missing summary=")
    elif not (
        isinstance(summary.value, ast.Constant)
        and isinstance(summary.value.value, str)
        and summary.value.value.strip()
    ):
        out.append(f"{loc}: summary= must be a non-empty string literal")

    if not ast.get_docstring(func):
        out.append(f"{loc}: route function is missing a docstring")

    path_arg = decorator.args[0] if decorator.args else None
    if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
        defaults = _arg_defaults(func)
        for pname in _PATH_TOKEN.findall(path_arg.value):
            if pname not in defaults:
                continue  # captured elsewhere (dependency / **kwargs) — don't guess
            dflt = defaults[pname]
            if not (
                isinstance(dflt, ast.Call)
                and isinstance(dflt.func, ast.Name)
                and dflt.func.id in PATH_PARAM_FUNCS
                and _has_description_kw(dflt)
            ):
                out.append(
                    f"{loc}: path param '{pname}' needs Path(description=...) "
                    f"(imported as PathParam where pathlib.Path is used)"
                )
    return out


def _route_decorator(func) -> Optional[ast.Call]:
    for dec in func.decorator_list:
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr in HTTP_METHODS
            and isinstance(dec.func.value, ast.Name)
        ):
            return dec
    return None


def _is_route_function(func) -> bool:
    return _route_decorator(func) is not None


def _resolve_str(node: ast.expr, consts: dict) -> Optional[str]:
    """Statically resolve a node to a string: a literal, a module-level string
    constant by name, or an `+`-concatenation of those. Returns None if it
    can't be resolved (f-strings, calls, unknown names). This lets the checker
    see `description=_SHARED + "…"` composed descriptions, not just literals."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve_str(node.left, consts)
        right = _resolve_str(node.right, consts)
        if left is not None and right is not None:
            return left + right
    return None


def _module_str_consts(tree: ast.Module) -> dict:
    """Map module-level `NAME = <string>` assignments (resolving concatenation
    and references to earlier constants), in source order."""
    consts: dict = {}
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            val = _resolve_str(stmt.value, consts)
            if val is not None:
                consts[stmt.targets[0].id] = val
    return consts


def _doc_string_nodes(tree: ast.Module):
    """User-facing doc text as (lineno, text) pairs: module/route/model-class
    docstrings and summary=/description= kwargs — including descriptions composed
    from module-level string constants (`_SHARED + "…"`)."""
    consts = _module_str_consts(tree)
    out = []
    models = _model_class_names(tree)

    def _docstring(node):
        b = getattr(node, "body", None)
        if (
            b
            and isinstance(b[0], ast.Expr)
            and isinstance(b[0].value, ast.Constant)
            and isinstance(b[0].value.value, str)
        ):
            out.append((b[0].value.lineno, b[0].value.value))

    _docstring(tree)  # module docstring
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in models:
            _docstring(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_route_function(node):
            _docstring(node)
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("summary", "description"):
                    text = _resolve_str(kw.value, consts)
                    if text is not None:
                        out.append((kw.value.lineno, text))
    return out


def _check_terminology(tree: ast.Module, rel: str) -> list[str]:
    out = []
    for lineno, text in _doc_string_nodes(tree):
        for pat, msg in _BANNED_TERMS:
            if pat.search(text):
                out.append(f"{rel}:{lineno}: banned term in doc text — {msg}")
        for pat, msg in _MECHANICS:
            if pat.search(text):
                out.append(f"{rel}:{lineno}: {msg}")
    return out


def find_violations(routers_dir: Path = ROUTERS_DIR) -> list[str]:
    violations: list[str] = []
    for path in sorted(routers_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = path.name
        tree = ast.parse(path.read_text(), filename=str(path))
        models = _model_class_names(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in models:
                violations.extend(_check_model_fields(node, rel))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                dec = _route_decorator(node)
                if dec is not None:
                    violations.extend(_check_route(node, dec, rel))
        violations.extend(_check_terminology(tree, rel))
    return violations


def main() -> int:
    violations = find_violations()
    if violations:
        print(f"API doc-style check failed ({len(violations)} violation(s)):\n")
        for v in violations:
            print(f"  - {v}")
        print(
            "\nSee the api-writing-style skill: "
            ".cursor/skills/api-writing-style/SKILL.md"
        )
        return 1
    print("API doc-style check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
