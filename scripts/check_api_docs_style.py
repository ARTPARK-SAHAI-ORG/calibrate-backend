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
    (
        re.compile(r"(?i)semantic categor(?:y|ies)"),
        "don't say 'semantic category' — describe it plainly (e.g. 'what the evaluator judges')",
    ),
    (
        re.compile(r"(?i)\bseeded\b"),
        "don't say 'seeded' — the reader-facing term for a shipped evaluator is "
        "'built-in default'",
    ),
    (
        re.compile(r"(?i)\bfree-?(?:text|form)\b"),
        "drop 'free-text' / 'free-form' — just describe the field (e.g. 'A description of the persona')",
    ),
    (
        # "Replacement X" as a noun-adjective prefix on an update field ("Replacement
        # persona set", "Replacement free-form payload"). The value IS the new value;
        # the word is filler. Describe it directly ("The item's new payload"). The
        # verb form ("Replaces the stored config") is a different word and stays fine.
        re.compile(r"(?i)\bReplacement\b"),
        "drop the filler 'Replacement' prefix — describe the value directly "
        "(e.g. 'The item's new payload'); the verb 'Replaces …' is fine",
    ),
    (
        re.compile(r"(?i)\b(?:creation|last[- ]update) timestamp\b|\btimestamp when\b"),
        "use the standard timestamp wording: 'When the <thing> was created / was "
        "last updated (ISO 8601 UTC)'",
    ),
    (
        re.compile(r"(?i)\bmedium\b"),
        "say 'modality' (text/audio), not 'medium'",
    ),
    (
        re.compile(r"(?i)\binlined\b|\bfor list views\b|\blist shape\b|\bdetail shape\b"),
        "don't describe the internal response shape ('inlined', 'for list views'); "
        "say what the value is to the reader",
    ),
    (
        re.compile(r"(?i)\b(?:\d+|zero|one)-(?:based|indexed)\b"),
        "don't say '1-based' / '0-indexed' (developer jargon); say it plainly, "
        "e.g. 'the first version is 1'",
    ),
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
        # Type+caveat parenthetical, e.g. "(object, optional)", "(array, required)",
        # "(string, optional)". This is the "(detail)" template the house style
        # bans: parentheses are for a bare unit/format/example only, never a
        # type-plus-required/optional gloss. Move required/optional into prose.
        re.compile(
            r"(?i)\((?:object|array|string|bool|boolean|int|integer|number"
            r"|dict|list|float)(?:\[[^\]]*\])?,\s*(?:required|optional)\)"
        ),
        "don't use a '(type, required/optional)' parenthetical (the '(detail)' "
        "template): say it in prose, e.g. 'the required conversation history'",
    ),
    (
        re.compile(r"(?i)human-readable"),  # filler
        "'Human-readable' is filler; drop it",
    ),
    (
        re.compile(r"(?i)hop-by-hop"),  # HTTP-plumbing jargon
        "don't say 'hop-by-hop' (HTTP-plumbing jargon); describe the effect for "
        "the reader, or drop it if it doesn't affect how they call the API",
    ),
    (
        # UI verb for API persistence. An API doesn't "save" — it creates, stores,
        # or persists. "before saving an agent", "a saved agent", "skip saving"
        # read like a form's Save button. Say "create" / "an existing agent" /
        # "stored". "stored config" already uses the allowed adjective.
        re.compile(r"(?i)\bsav(?:e|es|ed|ing)\b"),
        "don't use the UI verb 'save/saving/saved' for API persistence; say "
        "'create', 'store', 'persist', or 'an existing <thing>'",
    ),
    (
        # Adverb+participle compound modifier before a noun, e.g. "already-linked
        # tests", "newly-created rows". Rewrite as a relative clause: "tests that
        # are already linked". Domain terms like "soft-deleted" are intentionally
        # not matched (the adverb list is temporal-state only).
        re.compile(r"(?i)\b(?:already|newly|previously|recently|currently|freshly|just)-\w+"),
        "don't hyphenate an adverb onto a participle (e.g. 'already-linked'): "
        "rephrase as a relative clause, 'tests that are already linked'",
    ),
    (
        # Storage/DB implementation jargon leaking into user-facing text, e.g.
        # "link row IDs", "pinned on the pivot", "auto-increment link row ID".
        # Describe what the value means to the reader, not how it's stored. Bare
        # "row(s)" is fine ("max rows per eval"); only the storage-noun forms below.
        re.compile(r"(?i)\b(?:pivot|auto-increment|(?:link|join|db|table)[ -]rows?|row ids?)\b"),
        "don't expose storage jargon (pivot, link row, row id, auto-increment) in "
        "docs; describe the value's meaning (e.g. 'IDs of the links created')",
    ),
    (
        # Nullability caveats: the `| null` type is already shown, so ". null until
        # done", ". null if unset", ", or null" etc. are redundant. Matches only
        # caveat keywords after null, so INPUT-EFFECT phrasings survive ("null
        # clears the cell", "null to clear", "null marks a row-level overall").
        re.compile(
            r"(?i)(?:,\s*or\s+null\b"
            r"|[.]\s*`?null`?\s+(?:until|if|when|unless|for|on|before|while|unavailable)\b)"
        ),
        "drop the nullability caveat (the `| null` type already shows it); keep "
        "null only when it describes an input effect ('`null` clears the cell')",
    ),
    (
        # "per-X" compound modifier. Say "for each X". The lookbehind keeps this
        # from firing mid-token inside hyphenated code identifiers/paths like
        # `max-rows-per-eval` (there "per" is preceded by a hyphen, not prose).
        re.compile(r"(?i)(?<![\w-])per-[a-z]+"),
        "say 'for each X', not 'per-X' (e.g. 'results for each model', not "
        "'per-model results')",
    ),
    (
        # Validation caveats the API already enforces + returns a clear 400 for.
        # Describe what the field IS, not the empty-input rule ("Must be
        # non-empty", "(non-empty)"). This targets rendered field docs only —
        # runtime HTTPException error messages ("... must be non-empty") are not
        # scanned and stay, since they're the actual error feedback.
        re.compile(r"(?i)non-empty"),
        "drop 'non-empty' — the API rejects empty input with a 400 already; "
        "describe what the field holds",
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


def _trailing_literal(node: ast.expr) -> Optional[str]:
    """The statically-known trailing string of a description/summary value, or
    None when the tail can't be judged (ends in an f-string `{expr}`, a variable,
    etc.). Handles plain literals, f-strings (JoinedStr), and `+` concatenations —
    so `f"...{n}..batch."` and `CONST + "...tail."` are both inspected, not just
    bare `"..."` literals.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        last = node.values[-1] if node.values else None
        return last.value if isinstance(last, ast.Constant) and isinstance(last.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _trailing_literal(node.right)
    return None


def _check_trailing_periods(tree: ast.Module, rel: str) -> list[str]:
    """Flag `description=` / `summary=` values whose text ends in a single period.
    Param/field descriptions and summaries are terse labels in the house style,
    not sentences, so they carry no trailing period (internal sentence periods and
    `...` ellipses are fine). Docstrings are NOT checked here — they are prose and
    keep their terminal period. F-strings and concatenations are inspected via
    their trailing literal.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in ("description", "summary"):
                continue
            tail = _trailing_literal(kw.value)
            if tail is None:
                continue
            text = tail.rstrip()
            if text.endswith(".") and not text.endswith(".."):
                out.append(
                    f"{rel}:{node.lineno}: {kw.arg}= must not end with a period "
                    "(terse label, not a sentence); drop the trailing '.'"
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
        violations.extend(_check_trailing_periods(tree, rel))

    # Shared non-router modules that still produce rendered API doc text (e.g.
    # pagination.py builds Query() descriptions injected into list endpoints).
    # Only the `description=`/`summary=`-scoped check applies — their own module
    # and function docstrings are internal code prose that never renders as API
    # docs, so the docstring-based terminology pass is deliberately NOT run here.
    for extra in (REPO_ROOT / "src" / "pagination.py",):
        if not extra.exists():
            continue
        rel = str(extra.relative_to(REPO_ROOT))
        tree = ast.parse(extra.read_text(), filename=str(extra))
        violations.extend(_check_trailing_periods(tree, rel))
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
