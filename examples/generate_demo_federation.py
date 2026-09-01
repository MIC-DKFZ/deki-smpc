"""Generate credentials and environment files for the local MNIST demo.

This helper intentionally centralizes private keys for a single-machine demo.
Real participants generate and retain their identity private keys independently.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import shlex
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def env_value(value: str) -> str:
    return shlex.quote(value)


def write_secret(path: Path, values: dict[str, str]) -> None:
    content = "".join(f"{name}={env_value(value)}\n" for name, value in values.items())
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
    path.chmod(0o600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".demo-secrets"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--federation-id", default="mnist-demo")
    parser.add_argument("--clients", nargs="+", default=["site-a", "site-b", "site-c"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.clients) < 3 or len(set(args.clients)) != len(args.clients):
        raise SystemExit("provide at least three unique client IDs")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)

    private_keys: dict[str, str] = {}
    public_keys: dict[str, str] = {}
    participant_tokens: dict[str, str] = {}
    for client_id in args.clients:
        private_key = Ed25519PrivateKey.generate()
        private_keys[client_id] = encode(
            private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
        )
        public_keys[client_id] = encode(
            private_key.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
        )
        participant_tokens[client_id] = secrets.token_urlsafe(32)

    federation = {
        args.federation_id: {
            client_id: {
                "token": participant_tokens[client_id],
                "signing_public_key": public_keys[client_id],
                "role": "participant",
            }
            for client_id in args.clients
        }
    }
    common = {
        "DEKI_BASE_URL": args.base_url,
        "DEKI_FEDERATION_ID": args.federation_id,
    }
    admin_token = secrets.token_urlsafe(32)
    write_secret(
        args.output_dir / "server.env",
        {"DEKI_ADMIN_TOKEN": admin_token, "DEKI_FEDERATIONS_JSON": json.dumps(federation)},
    )
    # operator.env deliberately contains no participant private key or participant token.
    write_secret(
        args.output_dir / "operator.env",
        {**common, "DEKI_ADMIN_TOKEN": admin_token},
    )
    for client_id in args.clients:
        write_secret(
            args.output_dir / f"{client_id}.env",
            {
                **common,
                "DEKI_CLIENT_ID": client_id,
                "DEKI_AUTH_TOKEN": participant_tokens[client_id],
                "DEKI_IDENTITY_PRIVATE_KEY": private_keys[client_id],
                "DEKI_TRUSTED_SIGNING_KEYS": json.dumps(public_keys, sort_keys=True),
            },
        )

    manifest_path = args.output_dir / "public-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(public_keys, stream, indent=2, sort_keys=True)
        stream.write("\n")
    manifest_path.chmod(0o644)
    print(f"created local-demo credentials in {args.output_dir}")
    print(f"participants: {', '.join(args.clients)}")


if __name__ == "__main__":
    main()
