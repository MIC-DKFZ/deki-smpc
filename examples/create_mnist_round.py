"""Create a deki-smpc round whose schema matches the example MNIST model."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import httpx
from mnist_common import new_model

from deki_smpc.protocol.schema import ModelSchema


def env(name: str) -> str | None:
    return os.environ.get(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=env("DEKI_BASE_URL"))
    parser.add_argument("--admin-token", default=env("DEKI_ADMIN_TOKEN"))
    parser.add_argument("--federation-id", default=env("DEKI_FEDERATION_ID"))
    parser.add_argument("--participants", nargs="+", required=True)
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    parser.add_argument("--precision-bits", type=int, default=24)
    parser.add_argument("--protocol-version", choices=("1.0", "1.1"), default="1.1")
    parser.add_argument("--ca-bundle", default=env("DEKI_CA_BUNDLE"))
    parser.add_argument("--output", type=Path, help="also write the returned round ID to this file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name, value in {
            "--base-url/DEKI_BASE_URL": args.base_url,
            "--admin-token/DEKI_ADMIN_TOKEN": args.admin_token,
            "--federation-id/DEKI_FEDERATION_ID": args.federation_id,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"missing configuration: {', '.join(missing)}")
    if len(args.participants) < 3 or len(set(args.participants)) != len(args.participants):
        raise SystemExit("provide at least three unique participants")

    schema = ModelSchema.from_state_dict(new_model().state_dict(), precision_bits=args.precision_bits)
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/v1/federations/{args.federation_id}/rounds",
        headers={
            "Authorization": f"Bearer {args.admin_token}",
            "Idempotency-Key": f"mnist-round-{uuid.uuid4()}",
        },
        json={
            "protocol_version": args.protocol_version,
            "model_schema": schema.as_dict(),
            "model_schema_hash": schema.hash,
            "participants": args.participants,
            "deadline_seconds": args.deadline_seconds,
        },
        timeout=30,
        verify=args.ca_bundle or True,
    )
    response.raise_for_status()
    round_id = response.json()["round_id"]
    if args.output:
        args.output.write_text(f"{round_id}\n", encoding="utf-8")
    print(round_id)


if __name__ == "__main__":
    main()
