"""Pass only when a valid-but-modified aggregate is rejected."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

from deki_smpc import AggregateIntegrityError, FedAvgClient

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
before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
with FedAvgClient(
    base_url=os.environ["DEKI_BASE_URL"],
    federation_id="local-demo",
    client_id=client_id,
    auth_token=os.environ["DEKI_AUTH_TOKEN"],
    identity_private_key=os.environ["DEKI_IDENTITY_PRIVATE_KEY"],
    trusted_signing_keys=json.loads(os.environ["DEKI_TRUSTED_SIGNING_KEYS"]),
    allow_insecure_http=True,
) as client:
    try:
        client.aggregate(model=model, round_id=json.loads(round_file.read_text())[0], timeout=timedelta(seconds=90))
    except AggregateIntegrityError:
        pass
    else:
        raise RuntimeError("tampered aggregate was accepted")
for name, tensor in model.state_dict().items():
    torch.testing.assert_close(tensor, before[name])
Path(f"/coord/{client_id}.tamper-ok").write_text("detected\n")
