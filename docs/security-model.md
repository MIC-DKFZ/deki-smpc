# Security model

deki-smpc v1 provides confidential additive aggregation and client-side
detection of aggregate modification for a round-committed cross-silo
participant set.

## Actors

| Actor | Responsibility |
| --- | --- |
| Federation operator | Enroll participants, pin identity keys, create rounds, and distribute round IDs |
| Participant | Protect its identity key and token, verify the federation manifest, and execute the client protocol |
| Aggregation service | Authenticate, enforce barriers, store artifacts, and compute the aggregate |

The aggregation service is treated as an untrusted computational party for
model confidentiality and result integrity. Clients verify signed setup
material and the aggregate integrity equation locally.

## Assumptions

The v1 security argument uses these assumptions:

- Ed25519 signatures, X25519 key agreement, HKDF-SHA256, AES-256-GCM, and
  AES-256-CTR satisfy their standard cryptographic properties.
- Every participant starts with an authentic, complete Ed25519 public-key
  manifest.
- Participant hosts protect private identity keys, bearer tokens, ephemeral
  private keys, and decrypted round shares.
- TLS protects bearer credentials and request metadata in transit.
- Each round contains the complete committed participant set. Clients reject
  a round whose reported participant count differs from their pinned
  manifest, or that commits fewer than three participants.
- The aggregation service operates independently from participant hosts.
- The application releases aggregates for groups whose size and update
  distribution provide suitable privacy for the use case.

## Confidentiality

Under legacy protocol `1.0`, every update is encoded into the int64 ring and
masked with pairwise streams. Under default protocol `1.1`, it is masked with a
fresh private model key whose aggregate is built through blinded groups and an
encrypted binary tree. The service cannot decrypt the final `1.1` aggregate. In
`1.0`, one side of each participant pair adds a stream and the other subtracts
it, so the complete sum cancels all pairwise streams.

The server receives signed ephemeral public keys, encrypted share envelopes,
masked updates, individual padded tags, and the masked aggregate. Client
private keys, decrypted shares, pairwise mask keys, model keys, the common
integrity seed, and the aggregate field pad remain within participant
processes. The service learns the clear aggregate only for `1.0`.

The aggregate itself is an authorized protocol output. Federation governance
selects participant count, round frequency, model design, and any additional
privacy controls according to the information exposed by that output.

## Aggregate integrity

The server can apply additive changes to int64 ciphertexts. v1 detects such a
change with a random linear digest in the prime field `2**127 - 1`. The
coefficient seed and aggregate field pad are reconstructed by the clients from
encrypted per-round shares.

The result is accepted after verification of:

- HTTP content SHA-256;
- safetensors structure;
- tensor names, shapes, and encoded dtypes;
- aggregate prime-field tag.

An aggregate-tag mismatch raises `AggregateIntegrityError` before fixed-point
decoding and before the caller receives a state dictionary. The protocol
response to detected modification is local failure and round abort by the
application or operator.

## Availability and participant behavior

v1 uses complete-participation barriers. A missing participant leads to round
expiry or operator abort. This gives a simple and auditable relationship
between the committed manifest, mask cancellation, and released aggregate.

Participant authentication and signed key setup establish identity and message
origin. Federation enrollment, training governance, and input validation govern
the scientific validity of participant updates.

The protocol does not provide Byzantine agreement, malicious-update filtering,
or dropout recovery. The `1.1` blinded group ring has an adjacent-collusion
limitation: a participant's two ring neighbors can jointly isolate its model-key
contribution. Participant selection and collusion risk remain federation
governance responsibilities.

## Key lifecycle

Long-term Ed25519 keys identify federation members. X25519 private keys,
pairwise derived keys, integrity shares, model-key seeds, group blinds, and
generated masks are fresh for one round. The client consumes round key material
after successful completion.

Identity-key rotation creates a new federation manifest. Existing rounds retain
their original manifest hash and complete under that committed context.
