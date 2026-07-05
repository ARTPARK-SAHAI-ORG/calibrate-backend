"""patch-goreleaser-config.sh must fix Speakeasy's brews[].token indentation."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github/scripts/patch-goreleaser-config.sh"

BROKEN = """\
brews:
  - name: calibrate
    repository:
      owner: dalmia
      name: homebrew-tap
      branch: main
    token: "{{ .Env.HOMEBREW_TAP_GITHUB_TOKEN }}"
"""

FIXED = """\
brews:
  - name: calibrate
    repository:
      owner: dalmia
      name: homebrew-tap
      branch: main
      token: "{{ .Env.HOMEBREW_TAP_GITHUB_TOKEN }}"
"""


def _run_patch(content: str, tmp_path: Path) -> str:
    target = tmp_path / ".goreleaser.yaml"
    target.write_text(content)
    subprocess.run(
        ["bash", str(SCRIPT), str(target)],
        check=True,
        cwd=REPO_ROOT,
    )
    return target.read_text()


def test_moves_token_under_repository(tmp_path):
    assert _run_patch(BROKEN, tmp_path) == FIXED


def test_idempotent_on_already_patched(tmp_path):
    first = _run_patch(BROKEN, tmp_path)
    second = _run_patch(first, tmp_path)
    assert second == FIXED
