"""Guard: nothing calls the raw boto3 S3 client transfer methods directly.

When OBJECT_STORAGE_MODE=local, get_s3_client() returns None and artifacts live
on local disk. The helpers upload_file_to_s3 / download_file_from_s3 /
list_object_keys branch on is_local_object_storage(); a direct
`s3.download_file(...)` / `s3.get_paginator(...)` call crashes on the None
client (see architecture.md). The wrappers live in utils.py, so the raw methods
are allowed there and nowhere else.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# boto3 S3 client methods that bypass the local-storage branch.
BANNED = {
    "download_file",
    "upload_file",
    "download_fileobj",
    "upload_fileobj",
    "get_paginator",
}

# The wrappers that legitimately call the raw client.
ALLOWED_FILE = SRC / "utils.py"


def test_no_direct_s3_client_calls_outside_utils():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == ALLOWED_FILE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in BANNED
            ):
                offenders.append(
                    f"{path.relative_to(SRC.parent)}:{node.lineno} .{node.func.attr}()"
                )
    assert not offenders, (
        "Go through upload_file_to_s3 / download_file_from_s3 / list_object_keys, "
        "never the raw S3 client (breaks OBJECT_STORAGE_MODE=local). Offenders:\n"
        + "\n".join(offenders)
    )
