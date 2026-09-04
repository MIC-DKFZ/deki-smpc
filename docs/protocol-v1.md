# deki-smpc protocol v1

This document specifies the secure aggregation protocol implemented by
This document specifies the legacy protocol wire value `1.0`, preserved
unchanged in package release 1.0.1. New rounds use
[protocol 1.1](protocol-v1.1.md) unless an operator explicitly selects `1.0`.

## Protocol objective

A round-committed set of at least 3 participants computes an equal-weight
aggregate of integer-encoded model updates. The aggregation service receives
masked individual updates. Every participant verifies the final aggregate
before decoding it. Deployments may configure a participant limit; the
protocol itself does not define a maximum.

The round is the unit of state, authorization, key material, and retention.
Each round commits:

- protocol wire value;
- federation identifier and round identifier;
- ordered participant manifest and its SHA-256 hash;
- ordered model schema and its SHA-256 hash;
- fixed-point precision;
- tensor aggregation policies;
- deadline and retention time.

## Round state machine

The durable active states are:

```text
CREATED
  -> REGISTRATION_OPEN
  -> KEY_SETUP
  -> UPDATE_COLLECTION
  -> AGGREGATING
  -> RESULT_READY
  -> COMPLETED
```

`FAILED`, `ABORTED`, and `EXPIRED` are terminal states. Each state transition is
transactional and records an audit event. The server advances a phase after all
committed participants reach its barrier.

The default round deadline is 30 minutes. Operators may configure a deadline
from 5 seconds to 24 hours. The default retention interval is 24 hours.

## Model schema

The schema contains tensor name, shape, original dtype, and aggregation policy.
Tensor names follow Unicode code-point order. Canonical JSON uses UTF-8, sorted
object keys, and compact separators. SHA-256 of that encoding is the schema
hash.

v1 accepts up to 4,096 tensors, rank 8, dimensions up to 1,000,000,000, and one
artifact up to the configured server limit. The supplied server limit is 512
MiB.

The policies are:

| Policy | Definition |
| --- | --- |
| `MEAN` | Decode the summed value and divide by the committed participant count |
| `SUM` | Decode the summed value |
| `KEEP_LOCAL` | Preserve the local tensor value at each participant |

Floating-point tensors use `MEAN` by default. Integer tensors use `KEEP_LOCAL`
by default. Values selected for `MEAN` or `SUM` are finite floating-point
tensors.

## Fixed-point representation

Each selected value is multiplied by `2**precision_bits` and rounded to the
nearest integer with ties to even. The encoded value is stored as signed int64.
The client checks the available range with headroom for the full participant
count.

Masking and aggregation use two's-complement int64 addition. This operation is
the ring addition modulo `2**64` represented by signed int64 tensors.

## Authenticated key setup

Each federation member has a pinned Ed25519 identity key. For each round, every
participant creates a fresh X25519 key pair and signs the ephemeral public key.
Participants verify every signature against their local federation manifest.

The X25519 shared secret and HKDF-SHA256 derive domain-separated keys for:

- AES-256-GCM encryption of integrity shares;
- AES-256-CTR generation of model masks.

Each participant sends an encrypted seed share and an encrypted field-pad share
to every peer through the server. A signed key bundle binds all recipients and
ciphertexts to the complete round context. The server stores the signed public
material and encrypted envelopes.

After decryption, every participant computes the same key-context commitment.
The update phase begins when all commitments agree.

## Pairwise model masks

For each pair of participants, a domain-separated AES-256-CTR stream generates
signed int64 mask values in bounded-memory chunks. The participant with the
lexically lower identifier adds the stream. Its peer subtracts the same stream.
All pairwise masks therefore cancel in the sum of submitted updates.

Fresh ephemeral keys and round-context domain separation bind every mask to one
round, model schema, and participant manifest.

## Aggregate integrity

Aggregate integrity uses the prime field

```text
p = 2**127 - 1
```

Every participant reconstructs a common secret coefficient seed from the
received seed shares. It also reconstructs the aggregate field pad `Q` by
summing all field-pad shares modulo `p`. These values remain within the client
group.

A domain-separated AES-256-CTR generator produces field coefficients in
canonical tensor order. Participant `i` computes

```text
d_i = <r, x_i> mod p
tag_i = d_i + q_i mod p
```

where `x_i` is the encoded local update and `q_i` is its field-pad share. The
tag accompanies the masked update. The worker sums model tensors in the int64
ring and tags in the prime field.

After downloading the result, a participant computes the digest of the
aggregated encoded tensors and verifies

```text
<r, sum(x_i)> = sum(tag_i) - Q mod p
```

The comparison uses fixed-width 16-byte encodings and constant-time comparison.
Verification precedes fixed-point decoding and construction of the returned
state dictionary. A mismatch raises `AggregateIntegrityError`.

The integrity shares use the key-setup exchange, and the tag travels with the
model artifact. The round sequence therefore contains the same protocol phases
for model masking and aggregate verification.

## Retry and completion semantics

Every mutating request carries an `Idempotency-Key`. The durable idempotency
scope is client, round, and operation. A repeated key with identical content
returns the committed response. Reuse with different content returns
`IDEMPOTENCY_CONFLICT`.

Artifact slots are immutable. Client polling uses capped exponential backoff,
the server's `Retry-After` value, one total deadline, and an optional
cancellation callback.

Each client acknowledges completion after successful verification. The round
enters `COMPLETED` when every participant has acknowledged the result.

## Protocol scope

| Topic | v1 scope |
| --- | --- |
| Aggregation server modification | Detection followed by local round failure |
| Participant dropout | Round abort or expiry |
| Participant weighting | Equal weights |
| Malicious participant input | Operational enrollment and federation governance |
| Model privacy | Additive secure aggregation of committed updates |
| Privacy accounting | Application-level policy |
| Transport confidentiality | TLS deployment configuration |

The security assumptions and trust boundaries are specified in
[Security model](security-model.md).

## Stable error codes

The v1 error vocabulary includes:

- `AUTHENTICATION_FAILED`
- `AUTHORIZATION_FAILED`
- `PROTOCOL_VERSION_UNSUPPORTED`
- `ROUND_NOT_FOUND`
- `ROUND_CONFLICT`
- `IDEMPOTENCY_CONFLICT`
- `ROUND_EXPIRED`
- `ROUND_FAILED`
- `SCHEMA_MISMATCH`
- `ARTIFACT_INVALID`

The client maps these codes to the typed exceptions documented in
[Client API](client-api.md).
