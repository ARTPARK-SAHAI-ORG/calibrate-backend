"""Guard: the integration-testing fake must cover every place the real
``calibrate-agent`` is spawned.

Statically scans ``src/`` for every CLI invocation, extracts its subcommand, and
asserts the fake ([src/testing/fake_calibrate_agent.py]) handles it — so a new
real usage can't ship without a fake handler, and nobody has to hand-maintain a
list. Auto-captures new call sites; no edit to this test needed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import List, Set, Tuple

_SRC = Path(__file__).resolve().parents[1] / "src"
_CLI_FUNC = "get_calibrate_agent_cli"


def _load_supported_subcommands() -> Set[str]:
    """Import the standalone fake by path (it isn't a package import)."""
    fake_path = _SRC / "testing" / "fake_calibrate_agent.py"
    spec = importlib.util.spec_from_file_location("_fake_calibrate_agent", fake_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.SUPPORTED_SUBCOMMANDS)


def _is_cli_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _CLI_FUNC
    )


def _scan_file(path: Path) -> Tuple[Set[str], List[str]]:
    """Return (subcommands, unresolved) found in one source file.

    ``unresolved`` records ``file:line`` for any invocation whose subcommand
    couldn't be read as a string literal — the test fails on those so subcommands
    stay statically analyzable (and this scanner stays honest).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    # Variables bound to the CLI (e.g. `cli = get_calibrate_agent_cli()`), so
    # `[cli, "stt", ...]` is recognised just like an inline call.
    cli_vars: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_cli_call(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    cli_vars.add(tgt.id)

    def _is_cli_ref(node: ast.AST) -> bool:
        return _is_cli_call(node) or (
            isinstance(node, ast.Name) and node.id in cli_vars
        )

    subcommands: Set[str] = set()
    unresolved: List[str] = []

    def _take_next(seq: List[ast.AST], i: int, lineno: int) -> None:
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        if isinstance(nxt, ast.Constant) and isinstance(nxt.value, str):
            subcommands.add(nxt.value)
        else:
            unresolved.append(f"{path.relative_to(_SRC.parent)}:{lineno}")

    for node in ast.walk(tree):
        # `[cli, "<sub>", ...]` list literals (covers subprocess.Popen([...])).
        if isinstance(node, ast.List):
            for i, elt in enumerate(node.elts):
                if _is_cli_ref(elt):
                    _take_next(node.elts, i, elt.lineno)
        # `create_subprocess_exec("uv", "run", cli(), "<sub>", ...)` positional args.
        elif isinstance(node, ast.Call):
            for i, arg in enumerate(node.args):
                if _is_cli_ref(arg):
                    _take_next(node.args, i, arg.lineno)

    return subcommands, unresolved


def _scan_all() -> Tuple[Set[str], List[str]]:
    all_subs: Set[str] = set()
    all_unresolved: List[str] = []
    for path in _SRC.rglob("*.py"):
        if path.name == "fake_calibrate_agent.py":
            continue  # the fake defines subcommands; it isn't a call site
        subs, unresolved = _scan_file(path)
        all_subs |= subs
        all_unresolved += unresolved
    return all_subs, all_unresolved


def test_every_real_subcommand_is_handled_by_the_fake():
    used, unresolved = _scan_all()

    assert not unresolved, (
        "Found calibrate-agent invocations whose subcommand isn't a string "
        f"literal (make it one so it stays analyzable): {unresolved}"
    )

    # Sanity: the scanner actually found the known call sites. If this regresses
    # to empty, the scanner is broken and the subset check below is vacuous.
    baseline = {"llm", "stt", "tts", "simulations"}
    assert baseline <= used, (
        f"scanner found {sorted(used)} — expected at least {sorted(baseline)}; "
        "the AST scanner likely needs updating for a new call-site shape"
    )

    supported = _load_supported_subcommands()
    missing = used - supported
    assert not missing, (
        f"Real code spawns calibrate-agent subcommand(s) {sorted(missing)} that "
        "the fake doesn't handle. Add a handler in "
        "src/testing/fake_calibrate_agent.py (and its SUPPORTED_SUBCOMMANDS), "
        "plus an end-to-end case in tests/test_fake_calibrate_agent.py."
    )
