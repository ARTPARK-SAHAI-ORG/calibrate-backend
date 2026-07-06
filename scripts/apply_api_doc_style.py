#!/usr/bin/env python3
"""Apply mechanical api-writing-style fixes to router doc text.

Targets rendered docs only: route docstrings, summary=/description= kwargs,
and module-level docstrings. Skips code comments and non-route helper docstrings.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROUTERS = Path("src/routers")
HTTP = {"get", "post", "put", "patch", "delete"}

# Longest first.
SUBS = [
    ("workspace (workspace)", "workspace"),
    ("an workspace", "a workspace"),
    ("the caller's current workspace", "your workspace"),
    ("the caller's active workspace", "your workspace"),
    ("the caller's CURRENT WORKSPACE", "your workspace"),
    ("the caller's workspace", "your workspace"),
    ("caller's current workspace", "your workspace"),
    ("caller's active workspace", "your workspace"),
    ("caller's workspace", "your workspace"),
    ("The caller must be a member", "You must be a member"),
    ("the caller must be a member", "you must be a member"),
    ("with the caller as owner", "with you as owner"),
    ("caller-supplied config", "config you supply"),
    ("stores the caller config as-is", "stores the config you supply as-is"),
    ("when a caller omits this variable", "when you omit this variable"),
    ("(8-char identifier)", ""),
    ("(8-char UUID)", ""),
    ("(8-char)", ""),
    ("8-char UUID", "ID"),
    ("8-char identifier", "ID"),
    ("8-char identifiers", "IDs"),
    ("8-char) ", "ID) "),  # rare tail fragments
    (" by UUID.", " by ID."),
    (" by UUID ", " by ID "),
    (" to UUIDs", " to IDs"),
    (" to UUID", " to ID"),
    (" UUIDs ", " IDs "),
    (" UUID ", " ID "),
    (" UUID.", " ID."),
    (" UUID,", " ID,"),
    (" UUID)", " ID)"),
    ("UUID of ", "ID of "),
    ("UUID (", "ID ("),
    ("evaluator UUIDs", "evaluator IDs"),
    ("evaluator UUID", "evaluator ID"),
    ("dataset UUID", "dataset ID"),
    ("agent UUID", "agent ID"),
    ("job UUID", "job ID"),
    ("metric UUIDs", "metric IDs"),
    ("metric UUID", "metric ID"),
    ("UUIDs (IDs)", "IDs"),
    (" UUIDs.", " IDs."),
    (" UUID →", " ID →"),
    (" UUID;", " ID;"),
    # tighten doubled spaces from removals
]

_SPACE_RE = re.compile(r"  +")
_TRAIL_SPACE_PAREN = re.compile(r" \)")


def has_route_decorator(func) -> bool:
    for dec in func.decorator_list:
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr in HTTP
            and isinstance(dec.func.value, ast.Name)
        ):
            return True
    return False


def _model_class_names(tree: ast.Module) -> set[str]:
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


def target_nodes(tree: ast.Module):
    nodes = []
    models = _model_class_names(tree)
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        nodes.append(body[0].value)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in models:
            b = node.body
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                nodes.append(b[0].value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and has_route_decorator(node):
            b = node.body
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                nodes.append(b[0].value)
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("summary", "description") and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    nodes.append(kw.value)
    seen, out = set(), []
    for n in sorted(nodes, key=lambda x: (x.lineno, x.col_offset)):
        key = (n.lineno, n.col_offset)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def transform(text: str) -> str:
    for a, b in SUBS:
        text = text.replace(a, b)
    text = _SPACE_RE.sub(" ", text)
    text = _TRAIL_SPACE_PAREN.sub(")", text)
    text = text.replace(" .", ".")
    return text.strip() if text.startswith('"') else text  # preserve multiline indent


def process(path: Path) -> int:
    source = path.read_text()
    tree = ast.parse(source)
    result, cursor, changes = source, 0, 0
    for node in target_nodes(tree):
        seg = ast.get_source_segment(source, node)
        if seg is None:
            continue
        inner = node.value
        new_inner = transform(inner)
        if new_inner == inner:
            continue
        # Rebuild segment preserving quote style from AST literal repr
        new_seg = seg.replace(inner, new_inner, 1)
        idx = result.index(seg, cursor)
        result = result[:idx] + new_seg + result[idx + len(seg) :]
        changes += 1
        cursor = idx + len(new_seg)
    if changes:
        path.write_text(result)
    return changes


def main():
    total = 0
    for path in sorted(ROUTERS.glob("*.py")):
        if path.name == "__init__.py":
            continue
        c = process(path)
        if c:
            print(f"{path}: {c}")
            total += c
    print("TOTAL:", total)


if __name__ == "__main__":
    main()
