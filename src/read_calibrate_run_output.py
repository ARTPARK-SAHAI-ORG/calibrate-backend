"""Reads what a calibrate run left in its output directory and stdout: did it finish, how many tests got no answer, did it stop early, and which line says why it failed."""

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def unanswered_case_count(test_results: Optional[List[Dict[str, Any]]]) -> int:
    """How many parsed rows produced no answer. Used when calibrate's own count
    is not on disk yet, so a run in progress and a run that wrote no
    ``metrics.json`` still report their gaps."""
    return sum(1 for r in test_results or [] if r.get("unanswered"))



def cli_error_line(stdout: str, stderr: str, returncode: int) -> str:
    """The one line worth showing a reader when the eval tool wrote nothing."""
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    out = [l.strip() for l in ansi.sub("", stdout or "").splitlines() if l.strip()]
    err = [l.strip() for l in ansi.sub("", stderr or "").splitlines() if l.strip()]
    # calibrate prints the failure it stopped on as its last ❌ or ✗ line.
    for line in reversed(out):
        if line.startswith(("❌", "✗")):
            return line
    for lines in (out, err):
        for line in reversed(lines):
            if re.search(r"\berror\b", line, re.I):
                return line
    for lines in (err, out):
        if lines:
            return lines[-1]
    return "The eval tool stopped before it produced any result."


def run_counts(
    metrics_data: Optional[Dict[str, Any]], test_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """``unanswered_tests`` / ``stopped_early`` for a run's results, from metrics.json when present."""
    metrics_data = metrics_data or {}
    return {
        "unanswered_tests": (
            metrics_data["errored"]
            if metrics_data.get("errored") is not None
            else unanswered_case_count(test_results)
        ),
        "stopped_early": bool(metrics_data.get("stopped_early")),
    }


class CliRunFailed(subprocess.CalledProcessError):
    """A CLI run that left no finished results. ``error_line`` is what a reader may see."""

    error_line: str = ""
    log_message: str = ""


def no_output_failure(
    process, run_cmd, stdout: str, stderr: str, output_dir: Path, noun: str
) -> CliRunFailed:
    """The failure for a run that wrote no metrics.json. The caller logs ``log_message``."""
    if process.returncode != 0:
        error_line = cli_error_line(stdout, stderr, process.returncode)
        error_msg = f"{noun} failed with exit code {process.returncode}: {error_line}"
    else:
        # The path is for the log only; the reader sees the short line.
        error_line = "The eval tool produced no results."
        error_msg = f"{noun} produced no output files (results.json/metrics.json not found in {output_dir})"
    err = CliRunFailed(process.returncode, run_cmd, stdout, stderr)
    err.error_line = error_line
    err.log_message = error_msg
    return err
