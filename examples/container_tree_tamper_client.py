"""Pass only when a modified encrypted tree artifact cannot complete a round."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from deki_smpc import ArtifactValidationError, FedAvgClient, RoundFailedError, RoundTimeoutError

client_id = os.environ["DEKI_CLIENT_ID"]
round_file = Path("/coord/round_ids.json")
deadline = time.monotonic() + 60
while not round_file.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError("operator did not publish a round ID")
    time.sleep(0.1)
index = int(client_id.rsplit("-", 1)[1])
model = torch.nn.Linear(2, 1)
with torch.no_grad():
    model.weight.fill_(float(index))
    model.bias.fill_(float(index * 2))

with FedAvgClient(
    base_url=os.environ["DEKI_BASE_URL"],
    federation_id="local-demo",
    client_id=client_id,
    auth_token=os.environ["DEKI_AUTH_TOKEN"],
    identity_private_key=os.environ["DEKI_IDENTITY_PRIVATE_KEY"],
    trusted_signing_keys=json.loads(os.environ["DEKI_TRUSTED_SIGNING_KEYS"]),
    allow_insecure_http=True,
) as client:
    original_request = client.transport.request
    modified = False

    def tampering_request(self: object, method: str, path: str, **kwargs: Any) -> Any:
        global modified
        response = original_request(method, path, **kwargs)
        if not modified and method == "GET" and "/key-tasks/" in path and path.endswith("/artifact"):
            content = bytearray(response.content)
            content[0] ^= 1
            response._content = bytes(content)
            modified = True
        return response

    client.transport.request = MethodType(tampering_request, client.transport)  # type: ignore[method-assign]
    try:
        client.aggregate(
            model=model,
            round_id=json.loads(round_file.read_text())[0],
            timeout=timedelta(seconds=20),
        )
    except (ArtifactValidationError, RoundFailedError, RoundTimeoutError):
        pass
    else:
        raise RuntimeError("tree-artifact tampering did not stop aggregation")
Path(f"/coord/{client_id}.tree-tamper-ok").write_text("detected\n")
