# End-to-end MNIST federated training

This guide runs one real deki-smpc round with three local participants. Each
participant trains the same small PyTorch model on a different deterministic
shard of MNIST, uploads a masked model, verifies the aggregate, and saves the
new global model. The final section shows the small integration point to copy
into an existing training pipeline.

This is a functional local example, not a production deployment. Production
requires HTTPS, separately administered hosts, a secret manager, and an
appropriate federation governance and privacy policy.

## What the operator and sites exchange

There are two kinds of long-lived credentials. Do not confuse either one with
the fresh X25519 keys that the client creates automatically inside each round.

| Value | Created by | Given to | Purpose |
| --- | --- | --- | --- |
| `DEKI_ADMIN_TOKEN` | Federation operator | Server and operator only | Create and abort rounds |
| Participant bearer token | Federation operator | Server and that participant only | Authenticate one site |
| Ed25519 identity private key | Each participant | That participant only | Sign its per-round key setup |
| Ed25519 identity public key | Each participant | Operator, then every participant | Build the pinned federation manifest |
| `DEKI_FEDERATIONS_JSON` | Operator | Server only | Enroll IDs, tokens, roles, and public keys |
| `DEKI_TRUSTED_SIGNING_KEYS` | Operator from participant public keys | Every participant | Verify the complete participant set |
| Server URL and CA bundle | Service operator | Operator and every participant | Reach and authenticate the API |
| `round_id` | Server, when the operator creates a round | Every participant in that round | Select the committed aggregation round |

Tokens are generated secrets; there is no external key service from which they
are fetched. In production, use the organization's cryptographic random-secret
generator or secret manager. Each participant should generate its own Ed25519
key pair, retain the private key, and send only the raw public key to the
operator over an authenticated administrative channel. Both raw Ed25519 keys
are represented as padded standard Base64 strings by this client.

The complete public-key manifest must be identical at the server and every
site. A site must not build the manifest from values returned by the
aggregation server: the out-of-band pinned copy is what lets it detect a
substituted identity.

## 1. Prepare the repositories and environment

Place the client and server repositories next to one another:

```text
parent/
├── deki-smpc/
└── deki-smpc-server/
```

From `deki-smpc`, create one Python 3.12 or newer environment and install the
client, the MNIST dependency, and the sibling server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mnist]'
python -m pip install -e ../deki-smpc-server
```

In a real federation, each site installs only `deki-smpc` and its own training
dependencies. The service operator installs `deki-smpc-server`.

## 2. Generate local-demo credentials

For convenience, this demo creates all credentials on one machine:

```bash
python examples/generate_demo_federation.py \
  --clients site-a site-b site-c \
  --output-dir .demo-secrets
```

The generated directory contains:

```text
.demo-secrets/
├── server.env             # admin token and complete server enrollment
├── operator.env           # admin token, federation ID, and API URL
├── public-manifest.json   # safe-to-distribute public Ed25519 keys
├── site-a.env             # site-a token, private key, and public manifest
├── site-b.env
└── site-c.env
```

All `.env` files have mode `0600`, the directory is ignored by this
repository, and the generator refuses to overwrite a non-empty output
directory. Still, treat it as secret material and delete it when the demo is
finished. Never use this centralized generator to provision a production
federation; participant private keys should never pass through the operator.

## 3. Start the API and aggregation worker

The API and worker need the same enrollment and durable data paths. From the
`deki-smpc` directory, create the local data area:

```bash
mkdir -p .demo-server/artifacts
```

In terminal 1, start the API:

```bash
source .venv/bin/activate
set -a
source .demo-secrets/server.env
set +a
export DEKI_DATABASE_PATH="$PWD/.demo-server/metadata.sqlite3"
export DEKI_ARTIFACT_PATH="$PWD/.demo-server/artifacts"
python -m uvicorn app.main:app \
  --app-dir ../deki-smpc-server/key-aggregation-server \
  --host 127.0.0.1 --port 8080
```

In terminal 2, start the required aggregation worker with the same values:

```bash
source .venv/bin/activate
set -a
source .demo-secrets/server.env
set +a
export DEKI_DATABASE_PATH="$PWD/.demo-server/metadata.sqlite3"
export DEKI_ARTIFACT_PATH="$PWD/.demo-server/artifacts"
PYTHONPATH=../deki-smpc-server/key-aggregation-server \
  python -m app.worker.aggregation
```

Confirm readiness from another terminal:

```bash
curl --fail http://127.0.0.1:8080/health/ready
```

Plain HTTP is acceptable only for this loopback demo. See the server
repository's `docs/deployment.md` for durable storage, TLS ingress, process,
resource, and retention configuration.

## 4. Download MNIST and create a round

Download the dataset once before starting concurrent participants:

```bash
source .venv/bin/activate
python -c "from torchvision.datasets import MNIST; MNIST('mnist-data', train=True, download=True); MNIST('mnist-data', train=False, download=True)"
```

The operator creates a round from the exact model schema used at every site:

```bash
set -a
source .demo-secrets/operator.env
set +a
python examples/create_mnist_round.py \
  --participants site-a site-b site-c \
  --output .demo-round-id
```

The command prints and saves the server-generated `round_id`. In a real
workflow, the operator distributes that ID to the committed sites through the
federation's authenticated orchestration channel. The operator must create the
schema from the same state-dictionary names, shapes, dtypes, precision, and
tensor policies used by every participant.

## 5. Train and aggregate all three sites

The protocol has complete-participation barriers, so start all three commands
without waiting for the first to finish. These subshells simulate three sites
on one host:

```bash
(
  set -a; source .demo-secrets/site-a.env; set +a
  python examples/mnist_federated_client.py \
    --round-id "$(cat .demo-round-id)" --site-index 0 --site-count 3 \
    --allow-insecure-http
) &
(
  set -a; source .demo-secrets/site-b.env; set +a
  python examples/mnist_federated_client.py \
    --round-id "$(cat .demo-round-id)" --site-index 1 --site-count 3 \
    --allow-insecure-http
) &
(
  set -a; source .demo-secrets/site-c.env; set +a
  python examples/mnist_federated_client.py \
    --round-id "$(cat .demo-round-id)" --site-index 2 --site-count 3 \
    --allow-insecure-http
) &
wait
```

Each process uses every third training example, starting at its own index. It
trains locally for one epoch and then blocks inside `aggregate()` until all
three sites reach the protocol barriers. Expected progress is:

```text
SMPC: joined
SMPC: key_setup_complete
SMPC: update_uploaded
SMPC: completed
aggregated test accuracy: ...
saved verified aggregate to mnist-checkpoints/<site>-<round-id>.pt
```

All three output state dictionaries are the same verified mean of the locally
trained models. Each site receives a separate local copy; the server never
receives an unmasked individual model.

For a faster plumbing-only smoke test, add `--max-train-samples 512` to all
three commands. The resulting accuracy is not meaningful, but the complete
training and SMPC path is unchanged.

## 6. Run another federated epoch

A new training epoch uses a new SMPC round because cryptographic state is
round-local. Create another round with a different output file:

```bash
set -a
source .demo-secrets/operator.env
set +a
python examples/create_mnist_round.py \
  --participants site-a site-b site-c \
  --output .demo-round-id-2
```

Run the three participant commands again, this time adding each site's prior
verified checkpoint:

```text
--input-checkpoint mnist-checkpoints/<site>-<previous-round-id>.pt
```

This gives the normal federated loop:

```text
common global checkpoint
  -> local training at each site
  -> one newly created SMPC round
  -> verified mean returned to every site
  -> next common global checkpoint
```

Do not reuse a `round_id`, and do not let one site begin the next local epoch
from an unaggregated local model.

## Integrate the client into an existing pipeline

The framework-specific part is just the boundary after local training and
before the next global epoch:

```python
from datetime import timedelta

from deki_smpc import FedAvgClient

model.load_state_dict(global_checkpoint)
train_locally(model, local_training_data)

with FedAvgClient(
    base_url=settings.smpc_url,
    federation_id=settings.federation_id,
    client_id=settings.client_id,
    auth_token=secrets.participant_token,
    identity_private_key=secrets.ed25519_private_key_base64,
    trusted_signing_keys=settings.complete_public_key_manifest,
    ca_bundle=settings.federation_ca_bundle,
) as client:
    verified_global_state = client.aggregate(
        model=model,
        round_id=round_id_from_operator,
        timeout=timedelta(minutes=30),
    )

model.load_state_dict(verified_global_state)
save_checkpoint(model.state_dict())
```

`aggregate()` reads a detached copy and does not modify `model`. Load its
return value explicitly. By default, floating-point state-dictionary entries
use equal-weight `MEAN`, while integer entries use `KEEP_LOCAL`. If the model
has special buffers or counters, inspect the schema and set explicit
`tensor_policies`; every site and the operator must use the same policies.

The current protocol requires at least three committed participants, equal
weighting, and every participant to finish. A missing site causes expiry or an
operator abort rather than a partial aggregate. Use one active aggregation at
a time per `FedAvgClient`, set a timeout longer than local network and server
aggregation latency, and surface typed deki-smpc exceptions to the orchestration
layer.

## Production checklist

- Replace loopback HTTP with HTTPS and remove `--allow-insecure-http`.
- Set `DEKI_CA_BUNDLE` when the service certificate uses a private CA.
- Generate each Ed25519 identity at its participant site and protect the raw
  private key in that site's secret manager.
- Generate independent high-entropy operator and participant bearer tokens;
  rotate and distribute them through authenticated secret channels.
- Have every site verify the complete public manifest out of band before use.
- Keep the operator credential out of participant hosts and training jobs.
- Use the same model code, checkpoint lineage, schema precision, and tensor
  policies at every site.
- Arrange an authenticated channel for round IDs and global-epoch state.
- Size server memory and artifact storage for eight bytes per uploaded model
  element plus safetensors metadata and concurrent-round retention.
- Decide whether the aggregate itself needs additional privacy controls such
  as clipping or differential privacy; secure aggregation does not make the
  released aggregate private by itself.

For failures and cancellation behavior, see [Client API](client-api.md). For
the cryptographic trust boundary, see [Security model](security-model.md).
