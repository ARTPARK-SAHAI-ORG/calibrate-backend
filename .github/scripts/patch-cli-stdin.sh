#!/usr/bin/env bash
# Post-process Speakeasy CLI output: stop every command blocking forever on
# stdin. Speakeasy's HasStdinInput treats "stdin is not a character device" as
# "the caller piped a body in", then io.ReadAll's it. A coding agent or CI
# runner attaches a socket that carries nothing and is never closed, so the
# read never returns and the command hangs before it sends its request --- even
# on a GET with no body. This patch narrows the check to the two cases that
# really do carry a body: a named pipe, and a non-empty regular file.
set -euo pipefail

DIR="${1:-.speakeasy/out/cli}"
FLAGS_GO="$DIR/internal/flagutil/flags.go"

if [[ ! -f "$FLAGS_GO" ]]; then
  echo "No generated CLI at $DIR — skipping stdin patch"
  exit 0
fi

if grep -q 'os.ModeNamedPipe' "$FLAGS_GO"; then
  echo "CLI stdin already patched at $DIR"
  exit 0
fi

python3 - "$FLAGS_GO" <<'PY'
import shutil
import subprocess
import sys
from pathlib import Path

flags_go = Path(sys.argv[1])
text = flags_go.read_text()

old = "\treturn (stat.Mode() & os.ModeCharDevice) == 0\n"
new = (
    "\tmode := stat.Mode()\n"
    "\t// A pipe (producer | calibrate) carries a body and is closed by its writer.\n"
    "\tif mode&os.ModeNamedPipe != 0 {\n"
    "\t\treturn true\n"
    "\t}\n"
    "\t// A regular file (calibrate < body.json) carries one when it holds something.\n"
    "\tif mode.IsRegular() {\n"
    "\t\treturn stat.Size() > 0\n"
    "\t}\n"
    "\t// Anything else — a terminal, /dev/null, or the socket a coding agent or CI\n"
    "\t// runner attaches — carries no body. Reading a socket nobody writes to and\n"
    "\t// nobody closes blocks forever, so never read it.\n"
    "\treturn false\n"
)

n = text.count(old)
if n != 1:
    raise SystemExit(f"::warning::{flags_go}: expected exactly one anchor, found {n}")

flags_go.write_text(text.replace(old, new))

gofmt = shutil.which("gofmt")
if gofmt:
    subprocess.run([gofmt, "-w", str(flags_go)], check=False)

print(f"Patched CLI stdin detection in {flags_go}")
PY
