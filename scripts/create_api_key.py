import _bootstrap  # noqa: F401

"""Generate a service key; store only its hash in a private, gitignored file."""

import argparse
import hashlib
import os
from pathlib import Path
import secrets
from src.utils import save_json
from src.service_controls import read_principals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--principal", required=True)
    parser.add_argument("--resident", action="append", default=[])
    parser.add_argument("--admin", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".secrets/auth.json"))
    args = parser.parse_args()
    if not args.admin and not args.resident:
        parser.error("Non-admin principals require at least one --resident")
    existing = read_principals(str(args.output)) if args.output.exists() else []
    key = secrets.token_urlsafe(32)
    principal = {
        "id": args.principal,
        "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
        "residents": args.resident,
        "admin": args.admin,
    }
    save_json(
        args.output,
        {
            "principals": [item for item in existing if item["id"] != args.principal]
            + [principal]
        },
    )
    os.chmod(args.output, 0o600)
    read_principals(str(args.output))
    print("Store this key in your secret manager; it is displayed once:", key)
