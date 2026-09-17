"""Reads what a calibrate run left in its output directory and stdout: did it finish, how many tests got no answer, did it stop early, and which line says why it failed."""

import re
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def unanswered_case_count(test_results: Optional[List[Dict[str, Any]]]) -> int:
    """How many parsed rows produced no answer. Used when calibrate's own count
    is not on disk yet, so a run in progress and a run that wrote no
    ``metrics.json`` still report their gaps."""
    return sum(1 for r in test_results or [] if r.get("unanswered"))



def cli_error_line(stdout: str, stderr: str, returncode: int) -> str:
    """What calibrate itself said about a run that wrote nothing: its last
    ❌ or ✗ line, else its last line naming an error, else how the process
    ended. Nothing here is worded for a reader; the app puts its own sentence
    in front of it."""
    # Every terminal control sequence, not only colours: a progress bar clears
    # the line before printing, and that prefix would hide a ❌ line.
    ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    out = [l.strip() for l in ansi.sub("", stdout or "").splitlines() if l.strip()]
    err = [l.strip() for l in ansi.sub("", stderr or "").splitlines() if l.strip()]
    for line in reversed(out):
        if line.startswith(("❌", "✗")):
            return line
    for lines in (out, err):
        for line in reversed(lines):
            if re.search(r"\berror\b", line, re.I):
                return line
    if returncode < 0:
        # The OS names its signals; SIGTERM is what `kill` and a shutdown send.
        try:
            sig = signal.Signals(-returncode)
            # macOS appends ": 15" to the description; the number is already said.
            desc = (signal.strsignal(sig) or "").split(":")[0]
            ended = f"process killed by signal {-returncode} ({sig.name}: {desc})"
        except ValueError:
            ended = f"process killed by signal {-returncode}"
    else:
        ended = f"exited with code {returncode}"
    return f"calibrate-agent {ended}"


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
    """A CLI run that left no finished results. ``error_line`` is calibrate's own account of it."""

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
        error_line = "calibrate-agent exited with code 0 but wrote no results.json or metrics.json"
        error_msg = f"{noun} produced no output files (results.json/metrics.json not found in {output_dir})"
    err = CliRunFailed(process.returncode, run_cmd, stdout, stderr)
    err.error_line = error_line
    err.log_message = error_msg
    return err
