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
