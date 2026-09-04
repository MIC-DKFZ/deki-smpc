# Protocol 1.1: hardened binary key tree

Package release 1.0.1 adds wire value `1.1` without changing wire value `1.0`.
New rounds default to `1.1`; an operator may explicitly create `1.0` rounds for
rolling upgrades. Membership remains fixed, aggregation remains equal-weighted,
and every committed participant must finish before the deadline.

## State path

```text
CREATED -> REGISTRATION_OPEN -> KEY_SETUP -> KEY_AGGREGATION
        -> UPDATE_COLLECTION -> AGGREGATING -> RESULT_READY -> COMPLETED
```

Protocol `1.0` retains its direct `KEY_SETUP -> UPDATE_COLLECTION` transition.
Terminal `FAILED`, `ABORTED`, and `EXPIRED` behavior is unchanged.

## Canonical topology

After every signed X25519 ephemeral key is committed, each client and the server
form the same canonical manifest, ordered by client ID. The shuffle seed is:

```text
SHA-256(round_context.domain("tree-topology") || canonical_json(ephemeral_manifest))
```

A SHA-256 counter stream drives Fisher-Yates with rejection sampling. For `n`
participants, `ceil(n / 4)` balanced groups are formed. Groups contain three or
four sites, except that exactly five sites form one five-member group. The
ordered plan contains all group and tree tasks. Its identifier is SHA-256 of
compact, sorted-key UTF-8 JSON. Clients recompute the plan and compare the hash
in constant time before processing a task.

## Three stages

1. **Parallel blinded groups.** Each group coordinator expands its private model
   key `K_i` and an independent blind `B`, sends `K_i + B` around the group, and
   each member adds its own `K_i`. The coordinator removes `B`. All groups start
   independently.
2. **Binary reduction.** Group aggregates are added pairwise at each level. An
   odd node is carried through an explicit task. With `g` groups, the tree depth
   is `ceil(log2(g))`.
3. **Final distribution.** The root encrypts the aggregate key once with an
   AES-256 key derived from the common integrity seed in the distinct
   `final-group-key` domain. Every participant authenticates, decrypts, and
   acknowledges the same immutable artifact.

Every intermediate payload is a safetensors artifact encrypted end to end with
X25519, HKDF-SHA256, and AES-256-GCM. Associated data and the Ed25519 signature
bind the round context, plan hash, task ID, stage, level, sender, receiver,
dependencies, schema hash, nonce, ciphertext size, and SHA-256 digest. The
digest covers the nonce and GCM ciphertext body; the GCM tag authenticates that
digest through the associated data. Task slots are immutable.

Model-key seeds and group blinds are independent of integrity seed shares. Key
tensors are expanded in bounded chunks, combined in place, zeroed after use,
and never sent to the server in the clear.

## Masked model aggregation

Participant `i` uploads:

```text
C_i = encoded_update_i + K_i  (in the int64 two's-complement ring)
```

The worker computes `C = sum(C_i)`. Unlike `1.0`, this value remains masked on
the server. A client decrypts the distributed `K = sum(K_i)`, computes `C - K`,
then performs the existing constant-time prime-field integrity check before
fixed-point decoding. `MEAN`, `SUM`, and `KEEP_LOCAL` semantics are unchanged.

## Security boundary

Protocol `1.1` hides the final clear aggregate from the server as well as hiding
individual updates. It does not provide Byzantine agreement, malicious-update
filtering, or dropout recovery. A missing or rejected contribution fails or
expires the round. The blinded ring also has the standard adjacent-collusion
limitation: the two neighbors of a participant, if colluding, can isolate that
participant's model key contribution. Fixed membership and institutional
governance are therefore part of the threat model.
