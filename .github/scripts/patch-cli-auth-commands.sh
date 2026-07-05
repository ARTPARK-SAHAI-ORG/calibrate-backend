#!/usr/bin/env bash
# Speakeasy emits auth login/logout under the "auth" group. Expose top-level
# "calibrate login" / "calibrate logout" (whoami is already at root) and hide
# the nested duplicates for backward compatibility.
set -euo pipefail

DIR="${1:-.speakeasy/out/cli}"
ROOT_GO="$DIR/internal/cli/root.go"
AUTH_GO="$DIR/internal/cli/auth.go"
LOGIN_GO="$DIR/internal/cli/login.go"
LOGOUT_GO="$DIR/internal/cli/logout.go"

if [[ ! -f "$ROOT_GO" || ! -f "$AUTH_GO" ]]; then
  echo "No generated CLI at $DIR — skipping auth command patch"
  exit 0
fi

if grep -q 'initLoginCmd(rootCmd)' "$ROOT_GO"; then
  echo "CLI auth commands already patched at $DIR"
  exit 0
fi

python3 - "$ROOT_GO" "$AUTH_GO" "$LOGIN_GO" "$LOGOUT_GO" <<'PY'
import re
import sys
from pathlib import Path

root_go, auth_go, login_go, logout_go = map(Path, sys.argv[1:5])

root_text = root_go.read_text()
root_text, n = re.subn(
    r"(\tif err := initWhoamiCmd\(rootCmd\); err != nil \{\n"
    r"\t\treturn nil, fmt\.Errorf\(\"init whoami: %w\", err\)\n"
    r"\t\}\n)"
    r"(\tif err := initVersionCmd\(rootCmd\); err != nil \{\n)",
    r"\1"
    r"\tif err := initLoginCmd(rootCmd); err != nil {\n"
    r"\t\treturn nil, fmt.Errorf(" + '"init login: %w", err)\n'
    r"\t}\n"
    r"\tif err := initLogoutCmd(rootCmd); err != nil {\n"
    r"\t\treturn nil, fmt.Errorf(" + '"init logout: %w", err)\n'
    r"\t}\n"
    r"\2",
    root_text,
    count=1,
)
if n != 1:
    raise SystemExit(f"::warning::{root_go}: could not wire root login/logout commands")
root_go.write_text(root_text)

auth_text = auth_go.read_text()
auth_text, n_login = re.subn(
    r"authCmd\.AddCommand\(&cobra\.Command\{\n"
    r"\t\tUse:\s+\"login\",\n"
    r"\t\tShort: \"Interactively configure authentication credentials\",\n"
    r"\t\tLong: `Interactively configure authentication credentials for calibrate\.\n"
    r"Secret credentials are stored in the OS keychain when available,\n"
    r"with a config file fallback\.\n\n"
    r"All fields are optional — press Enter to skip any field you don't need\.\n"
    r"Use the configure command for both authentication and global parameters\.`,\n"
    r"\t\tRunE: runAuthLoginCmd,\n"
    r"\t\}\)",
    """loginCmd := &cobra.Command{
\t\tUse:   "login",
\t\tShort: "Interactively configure authentication credentials",
\t\tLong: `Interactively configure authentication credentials for calibrate.
Secret credentials are stored in the OS keychain when available,
with a config file fallback.

All fields are optional — press Enter to skip any field you don't need.
Use the configure command for both authentication and global parameters.`,
\t\tRunE: runAuthLoginCmd,
\t}
\tloginCmd.Hidden = true
\tauthCmd.AddCommand(loginCmd)""",
    auth_text,
    count=1,
)
auth_text, n_logout = re.subn(
    r"authCmd\.AddCommand\(&cobra\.Command\{\n"
    r"\t\tUse:\s+\"logout\",\n"
    r"\t\tShort: \"Clear all stored authentication credentials\",\n"
    r"\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file\.\n\n"
    r"This removes all credentials previously set via auth login or configure\.`,\n"
    r"\t\tRunE: runAuthLogoutCmd,\n"
    r"\t\}\)",
    """logoutCmd := &cobra.Command{
\t\tUse:   "logout",
\t\tShort: "Clear all stored authentication credentials",
\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file.

This removes all credentials previously set via login or configure.`,
\t\tRunE: runAuthLogoutCmd,
\t}
\tlogoutCmd.Hidden = true
\tauthCmd.AddCommand(logoutCmd)""",
    auth_text,
    count=1,
)
if n_login != 1 or n_logout != 1:
    raise SystemExit(
        f"::warning::{auth_go}: could not hide nested auth login/logout "
        f"(login={n_login}, logout={n_logout})"
    )
auth_go.write_text(auth_text)

login_go.write_text(
    """// Patched by patch-cli-auth-commands.sh — top-level login shortcut.

package cli

import (
\t"github.com/spf13/cobra"
)

func initLoginCmd(parent *cobra.Command) error {
\tcmd := &cobra.Command{
\t\tUse:   "login",
\t\tShort: "Interactively configure authentication credentials",
\t\tLong: `Interactively configure authentication credentials for calibrate.
Secret credentials are stored in the OS keychain when available,
with a config file fallback.

All fields are optional — press Enter to skip any field you don't need.
Use the configure command for both authentication and global parameters.`,
\t\tRunE: runAuthLoginCmd,
\t}
\tparent.AddCommand(cmd)
\treturn nil
}
"""
)

logout_go.write_text(
    """// Patched by patch-cli-auth-commands.sh — top-level logout shortcut.

package cli

import (
\t"github.com/spf13/cobra"
)

func initLogoutCmd(parent *cobra.Command) error {
\tcmd := &cobra.Command{
\t\tUse:   "logout",
\t\tShort: "Clear all stored authentication credentials",
\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file.

This removes all credentials previously set via login or configure.`,
\t\tRunE: runAuthLogoutCmd,
\t}
\tparent.AddCommand(cmd)
\treturn nil
}
"""
)

print(f"Patched CLI auth commands in {root_go.parent.parent.parent}")
PY
