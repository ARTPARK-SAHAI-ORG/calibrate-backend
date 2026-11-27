"""Guard: the "calibrate-agent" binary name is a literal only in utils.py.

get_calibrate_agent_cli() is the single place that resolves the offline CLI
path. Under FAKE_AI_PROVIDERS it returns the in-repo fake instead of the real
binary, so every spawn must route through it (see architecture.md). Hardcoding
"calibrate-agent" anywhere else bypasses the test-mode seam and would run the
real CLI during tests. This complements test_fake_matches_real_usage.py, which
checks the subcommands but not the binary-name indirection.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
BINARY = "calibrate-agent"
ALLOWED_FILE = SRC / "utils.py"


def test_calibrate_agent_literal_only_in_utils():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == ALLOWED_FILE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == BINARY:
                offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")
    assert not offenders, (
        'Spawn the offline CLI only through get_calibrate_agent_cli(); do not '
        'hardcode "calibrate-agent" (breaks the FAKE_AI_PROVIDERS test seam). '
        "Offenders:\n" + "\n".join(offenders)
    )
