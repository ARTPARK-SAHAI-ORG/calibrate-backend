#!/usr/bin/env python3
"""Ensure every Public API route documents standard error responses."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTERS_DIR = REPO_ROOT / "src" / "routers"


def find_violations(routers_dir: Path = ROUTERS_DIR) -> list[str]:
    violations: list[str] = []
    for path in sorted(routers_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                if not _is_router_method_call(dec):
                    continue
                if not _has_public_api_tag(dec):
                    continue
                if not _has_responses_kw(dec, name="PUBLIC_API_ERROR_RESPONSES"):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(
                        f"{rel}:{node.lineno} {node.name} — Public API route missing "
                        "responses=PUBLIC_API_ERROR_RESPONSES"
                    )
    return violations


def _is_router_method_call(dec: ast.Call) -> bool:
    func = dec.func
    return isinstance(func, ast.Attribute) and func.attr in {
        "get",
        "post",
        "put",
        "patch",
        "delete",
    }


def _has_public_api_tag(dec: ast.Call) -> bool:
    for kw in dec.keywords:
        if kw.arg != "tags":
            continue
        if isinstance(kw.value, ast.List):
            for elt in kw.value.elts:
                if isinstance(elt, ast.Constant) and elt.value == "Public API":
                    return True
    return False


def _has_responses_kw(dec: ast.Call, *, name: str) -> bool:
    for kw in dec.keywords:
        if kw.arg != "responses":
            continue
        if isinstance(kw.value, ast.Name) and kw.value.id == name:
            return True
    return False


def main() -> int:
    violations = find_violations()
    if violations:
        print("Public API error-response documentation violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("All Public API routes document standard error responses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
