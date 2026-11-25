"""How much OpenRouter credit is left on the server's key.

Run from the repository root:

    uv run python scripts/openrouter_credit.py

Reads OPENROUTER_API_KEY from the environment, or from src/.env when it is
not set. Prints the key's own spending limit and what is left of it, then the
account's total credit, when the key is allowed to read it.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    env_file = Path(__file__).resolve().parent.parent / "src" / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip("'\"")
    return key


def get(path: str, key: str) -> dict | None:
    request = urllib.request.Request(
        f"https://openrouter.ai/api/v1{path}",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)["data"]
    except urllib.error.HTTPError as error:
        print(f"  {path}: OpenRouter answered {error.code}")
        return None


def dollars(value) -> str:
    return "no limit" if value is None else f"${value:,.2f}"


def main() -> int:
    key = api_key()
    if not key:
        print("OPENROUTER_API_KEY is not set, and src/.env does not have it.")
        return 1

    print("This key")
    data = get("/key", key)
    if data:
        print(f"  Name:            {data.get('label') or '(none)'}")
        print(f"  Spending limit:  {dollars(data.get('limit'))}")
        print(f"  Used:            {dollars(data.get('usage'))}")
        print(f"  Left:            {dollars(data.get('limit_remaining'))}")

    print("Whole account")
    data = get("/credits", key)
    if data:
        bought = data.get("total_credits") or 0
        used = data.get("total_usage") or 0
        print(f"  Credit bought:   {dollars(bought)}")
        print(f"  Used:            {dollars(used)}")
        print(f"  Left:            {dollars(bought - used)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
