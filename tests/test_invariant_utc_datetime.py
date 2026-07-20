"""Guard: the codebase never calls naive `datetime.now()`.

SQLite stores UTC and compares against `CURRENT_TIMESTAMP`. A bare
`datetime.now()` returns the server's local time, so a job's inactivity-timeout
math silently drifts by the machine's UTC offset (see architecture.md). The safe
forms are `datetime.utcnow()` or `datetime.now(timezone.utc)`; only the
argument-less `.now()` is a footgun. This fails CI instead of shipping the bug.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _naive_now_calls(tree: ast.AST) -> list[int]:
    """Line numbers of argument-less `datetime.now()` calls (no tz => local time)."""
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and not node.args and not node.keywords):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "now"):
            continue
        base = func.value
        is_datetime = (isinstance(base, ast.Name) and base.id == "datetime") or (
            isinstance(base, ast.Attribute) and base.attr == "datetime"
        )
        if is_datetime:
            hits.append(node.lineno)
    return hits


def test_no_naive_datetime_now_in_src():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line in _naive_now_calls(tree):
            offenders.append(f"{path.relative_to(SRC.parent)}:{line}")
    assert not offenders, (
        "Use datetime.utcnow() or datetime.now(timezone.utc), never a naive "
        "datetime.now() (SQLite stores UTC). Offenders:\n" + "\n".join(offenders)
    )
