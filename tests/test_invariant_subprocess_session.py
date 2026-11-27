"""Guard: every subprocess.Popen passes start_new_session=True.

The flag puts the child in its own process group so the job's abort/timeout path
can kill the whole tree with os.killpg (PID == PGID). Without it, aborting a job
leaves the calibrate subprocess alive and still writing output files into a job
the API has given up on (see architecture.md). Every runner already sets it;
this fails CI if a new Popen forgets.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _popen_without_session(tree: ast.AST) -> list[int]:
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_popen = (isinstance(func, ast.Attribute) and func.attr == "Popen") or (
            isinstance(func, ast.Name) and func.id == "Popen"
        )
        if not is_popen:
            continue
        has_flag = any(
            kw.arg == "start_new_session"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
        if not has_flag:
            hits.append(node.lineno)
    return hits


def test_every_popen_starts_new_session():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line in _popen_without_session(tree):
            offenders.append(f"{path.relative_to(SRC.parent)}:{line}")
    assert not offenders, (
        "subprocess.Popen must pass start_new_session=True so the job can be "
        "killed as a process group (see architecture.md). Offenders:\n"
        + "\n".join(offenders)
    )
