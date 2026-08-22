#!/usr/bin/env python3
"""Local stand-in for Codecov's `patch` check: are the lines this branch adds
covered by tests?

Codecov's patch target is `auto`, i.e. the base commit's coverage, so there is
no fixed number to hardcode. This run's own overall coverage stands in for the
base: main is nearly all of the code, so the two sit within a point of each
other.

Reads the `.coverage` file pytest already wrote — run the suite first.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def base_ref() -> str | None:
    for ref in ("origin/main", "main"):
        try:
            return git("merge-base", ref, "HEAD").strip()
        except subprocess.CalledProcessError:
            continue
    return None


def changed_lines(base: str) -> dict[str, set[int]]:
    """Lines this branch adds, per file."""
    diff = git("diff", "-U0", f"{base}...HEAD", "--", "src")
    changed: dict[str, set[int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path == "/dev/null":
                path = None
        elif line.startswith("@@") and path:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if not m:
                continue
            start = int(m.group(1))
            count = 1 if m.group(2) is None else int(m.group(2))
            changed.setdefault(path, set()).update(range(start, start + count))
    return changed


def main() -> int:
    base = base_ref()
    if not base:
        print("No main branch to compare against — skipping patch coverage.")
        return 0

    if not (ROOT / ".coverage").exists():
        print(
            "No .coverage found. Run `uv run --group dev pytest` first.",
            file=sys.stderr,
        )
        return 1

    from coverage import Coverage

    cov = Coverage(data_file=str(ROOT / ".coverage"))
    cov.load()

    overall_total = overall_missing = 0
    analyses: dict[str, tuple[set[int], set[int]]] = {}
    for measured in cov.get_data().measured_files():
        try:
            _, statements, _, missing, _ = cov.analysis2(measured)
        except Exception:
            continue
        overall_total += len(statements)
        overall_missing += len(missing)
        try:
            rel = str(Path(measured).resolve().relative_to(ROOT))
        except ValueError:
            continue
        analyses[rel] = (set(statements), set(missing))

    uncovered: list[str] = []
    patch_total = patch_hit = 0
    for path, lines in changed_lines(base).items():
        analysis = analyses.get(path)
        if not analysis:
            continue
        statements, missing = analysis
        for line in sorted(lines & statements):
            patch_total += 1
            if line in missing:
                uncovered.append(f"{path}:{line}")
            else:
                patch_hit += 1

    if patch_total == 0:
        print("No changed lines under src/ are measured — nothing to check.")
        return 0

    pct = lambda hit, total: 100.0 if total == 0 else hit / total * 100
    patch_pct = pct(patch_hit, patch_total)
    overall_pct = pct(overall_total - overall_missing, overall_total)

    print(
        f"Changed lines covered: {patch_hit}/{patch_total} ({patch_pct:.2f}%)\n"
        f"Target (this run's overall coverage): {overall_pct:.2f}%"
    )

    if patch_pct + 1e-9 >= overall_pct:
        print("Patch coverage is at or above the target.")
        return 0

    print(
        f"\nPatch coverage is below the target. {len(uncovered)} changed line(s) have no test:\n"
        + "\n".join(f"  {line}" for line in uncovered)
        + "\n\nAdd tests that reach these lines, or push with SKIP_COVERAGE=1 to bypass.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
