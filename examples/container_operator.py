"""Create the deterministic local Compose smoke-test round."""

import os
import time
import uuid
from pathlib import Path

import httpx
import torch

from deki_smpc.protocol.schema import ModelSchema

base_url = os.environ["DEKI_BASE_URL"]
deadline = time.monotonic() + 60
while True:
    try:
        if httpx.get(f"{base_url}/health/ready", timeout=2).status_code == 200:
            break
    except httpx.HTTPError:
        pass
    if time.monotonic() >= deadline:
        raise RuntimeError("API did not become ready")
    time.sleep(0.25)

schema = ModelSchema.from_state_dict(torch.nn.Linear(2, 1).state_dict())
round_ids = []
for _ in range(int(os.environ.get("DEKI_ROUND_COUNT", "5"))):
    response = httpx.post(
        f"{base_url}/v1/federations/local-demo/rounds",
        headers={
            "Authorization": f"Bearer {os.environ['DEKI_ADMIN_TOKEN']}",
            "Idempotency-Key": f"compose-{uuid.uuid4()}",
        },
        json={
            "protocol_version": "1.1",
            "model_schema": schema.as_dict(),
            "model_schema_hash": schema.hash,
            "participants": ["client-1", "client-2", "client-3"],
            "deadline_seconds": 120,
        },
        timeout=10,
    )
    response.raise_for_status()
    round_ids.append(response.json()["round_id"])
Path("/coord/round_ids.json").write_text(__import__("json").dumps(round_ids))
