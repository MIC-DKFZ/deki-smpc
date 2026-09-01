"""One independent Compose smoke-test participant."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

from deki_smpc import FedAvgClient

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
    for round_id in json.loads(round_file.read_text()):
        result = client.aggregate(model=model, round_id=round_id, timeout=timedelta(seconds=90))
        torch.testing.assert_close(result["weight"], torch.full((1, 2), 2.0))
        torch.testing.assert_close(result["bias"], torch.tensor([4.0]))
Path(f"/coord/{client_id}.ok").write_text("verified\n")
